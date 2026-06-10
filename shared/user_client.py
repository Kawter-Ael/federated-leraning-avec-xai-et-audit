from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from explainability.shap_explainer import compute_client_safe_explanations
from shared.federated_data import (
    build_artifact_paths,
    get_model_input_schema,
    load_preprocessing_metadata,
    load_structured_dataset,
)
from shared.modeling import (
    evaluate_model,
    predict_probabilities,
    select_decision_threshold,
    train_local_model,
)
from shared.utils import PipelineTracer, find_free_port

_logger = logging.getLogger("shared.user_client")


def _read_latest_trace_event(trace_path: Path) -> dict[str, Any] | None:
    if not trace_path.exists():
        return None
    lines = [
        line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _emit_progress_from_traces(
    *,
    artifact_root: str,
    global_artifact_root: str,
    progress_callback: Any | None,
) -> None:
    if progress_callback is None:
        return
    local_event = _read_latest_trace_event(Path(artifact_root) / "pipeline_trace.jsonl")
    global_event = _read_latest_trace_event(
        Path(global_artifact_root) / "pipeline_trace.jsonl"
    )
    latest = global_event or local_event
    if latest is None:
        return
    phase = str(latest.get("phase", ""))
    if phase == "server_launch":
        progress_callback("server_start", "Starting Flower server...")
    elif phase in {"client_training", "training_full"}:
        progress_callback("training", "Federated training in progress...")
    elif phase == "phase4_xai":
        progress_callback("xai", "Generating XAI artifacts...")
    elif phase == "phase5_audit":
        progress_callback("audit", "Running audit...")
    elif phase == "server_shutdown":
        progress_callback("finalize", "Finalizing training run...")


def prepare_data(user_context: dict[str, Any]) -> dict[str, Any]:
    artifact_root = str(user_context["artifact_root"])
    local_paths = build_artifact_paths(artifact_root)

    required = [
        local_paths["metadata"],
        local_paths["train"],
        local_paths["validation"],
        local_paths["test"],
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Phase 2 artifacts missing for this run — rerun preprocessing. "
            f"Missing files: {', '.join(missing)}"
        )

    metadata = load_preprocessing_metadata(artifact_root=artifact_root)
    schema = get_model_input_schema(metadata)
    return {
        "artifact_root": artifact_root,
        "metadata": metadata,
        "schema": schema,
        "train_dataset": load_structured_dataset(local_paths["train"]),
        "validation_dataset": load_structured_dataset(local_paths["validation"]),
        "test_dataset": load_structured_dataset(local_paths["test"]),
        "raw_shape": metadata.get("raw_shape", [0, 0]),
    }


def train_model(
    model: Any,
    prepared_data: dict[str, Any],
    run_config: dict[str, Any],
) -> Any:
    train_dataset = prepared_data["train_dataset"]
    train_local_model(
        model,
        train_dataset.x_numeric,
        train_dataset.x_categorical,
        train_dataset.y,
        run_config,
    )
    model.eval()
    return model


def compute_metrics(
    model: Any,
    prepared_data: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    validation_dataset = prepared_data["validation_dataset"]
    test_dataset = prepared_data["test_dataset"]
    train_dataset = prepared_data["train_dataset"]
    validation_probabilities = predict_probabilities(
        model,
        validation_dataset.x_numeric,
        validation_dataset.x_categorical,
    )
    # Training probabilities serve as fallback for threshold selection when
    # the validation split happens to be single-class (common on small,
    # severely imbalanced datasets after stratified split).
    train_probabilities = predict_probabilities(
        model,
        train_dataset.x_numeric,
        train_dataset.x_categorical,
    )
    selected_threshold, threshold_metrics = select_decision_threshold(
        validation_dataset.y,
        validation_probabilities,
        candidate_thresholds=run_config["training"].get("threshold_candidates"),
        selection_strategy=str(
            run_config["training"].get("threshold_selection_strategy", "f1")
        ),
        min_recall=float(
            run_config["training"].get("threshold_selection_min_recall", 0.0)
        ),
        fallback_y_true=train_dataset.y,
        fallback_probabilities=train_probabilities,
    )
    validation_evaluation = evaluate_model(
        model,
        validation_dataset.x_numeric,
        validation_dataset.x_categorical,
        validation_dataset.y,
        threshold=selected_threshold,
    )
    test_evaluation = evaluate_model(
        model,
        test_dataset.x_numeric,
        test_dataset.x_categorical,
        test_dataset.y,
        threshold=selected_threshold,
    )
    return {
        "selected_threshold": float(selected_threshold),
        "selection_metrics": threshold_metrics,
        "validation_metrics": {
            "loss": validation_evaluation.loss,
            **validation_evaluation.metrics,
        },
        "test_metrics": {"loss": test_evaluation.loss, **test_evaluation.metrics},
    }


def generate_shap_summary(
    model: Any,
    prepared_data: dict[str, Any],
    run_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    explainability_config = run_config.get("local_explainability", {})
    rules_config = run_config.get("local_rules", {})
    return compute_client_safe_explanations(
        model,
        prepared_data["metadata"],
        prepared_data["schema"],
        prepared_data["train_dataset"],
        background_size=int(explainability_config.get("background_size", 5)),
        sample_size=int(explainability_config.get("sample_size", 3)),
        nsamples=int(explainability_config.get("nsamples", 20)),
        top_features=int(explainability_config.get("top_features", 10)),
        instance_explanations=int(
            explainability_config.get("instance_explanations", 0)
        ),
        max_rules=int(rules_config.get("max_rules", 5)),
        min_support=float(rules_config.get("min_support", 0.05)),
        min_category_count=int(rules_config.get("min_category_count", 10)),
    )


def generate_rule_summary(local_rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rules": local_rules,
        "num_rules": len(local_rules),
        "aggregation": "client_local_rule_summary",
    }


# ---------------------------------------------------------------------------
# Docker-per-user helpers
# ---------------------------------------------------------------------------

def _to_host_path(container_path: str) -> str:
    """Convert a container-internal artifact path to its HOST equivalent.

    Inside the portal container `/app/artifacts` is mounted from
    `FL_ARTIFACT_ROOT_HOST` on the host.  Paths may arrive as either:
      - absolute:  /app/artifacts/user_runs/<run_id>
      - relative:  artifacts/user_runs/<run_id>   (Path("artifacts") / ...)

    Both forms are translated so that `docker run -v` gets an absolute
    host path.
    """
    host_base = os.environ.get("FL_ARTIFACT_ROOT_HOST") or str(Path.cwd() / "artifacts")
    path_str = str(container_path)

    for prefix in ("/app/artifacts", "artifacts"):
        if path_str == prefix:
            return host_base
        if path_str.startswith(prefix + "/") or path_str.startswith(prefix + "\\"):
            relative = path_str[len(prefix):].lstrip("/\\")
            return str(Path(host_base) / relative)

    return path_str


def _build_docker_client_cmd(
    *,
    client_id: str,
    server_address: str,
    artifact_root: str,
    global_artifact_root: str,
    config_override_json: str,
    log_dir: Path,
) -> list[str]:
    """Return the `docker run` command that starts one FL client container.

    The container image is ``ensaj-fl-client`` (or the value of the
    ``FL_DOCKER_IMAGE`` env var).  The container joins ``fl-net`` (or the
    value of ``FL_DOCKER_NETWORK``), which allows it to reach the portal
    container where the FL server subprocess is listening.

    Volume paths are translated from container-internal paths to HOST paths
    via ``_to_host_path``.
    """
    image = os.environ.get("FL_DOCKER_IMAGE", "ensaj-fl-client")
    network = os.environ.get("FL_DOCKER_NETWORK", "ensaj_fl-net")
    host_data = os.environ.get("FL_DATA_ROOT_HOST") or str(Path.cwd() / "data")
    host_config = os.environ.get("FL_CONFIG_ROOT_HOST") or str(Path.cwd() / "config")

    host_client_root = _to_host_path(artifact_root)
    host_global_root = _to_host_path(global_artifact_root)

    # Unique container name avoids conflicts between concurrent runs
    run_suffix = Path(artifact_root).name[-12:]
    container_name = f"ensaj-fl-{client_id}-{run_suffix}"

    cmd = [
        "docker", "run",
        "--rm",
        "--name", container_name,
        "--network", network,
        # Env vars
        "-e", f"CLIENT_ID={client_id}",
        "-e", f"FL_SERVER_ADDRESS={server_address}",
        "-e", f"CLIENT_ARTIFACT_ROOT=/app/client-data",
        "-e", f"GLOBAL_ARTIFACT_ROOT=/app/global-artifacts",
        "-e", f"CONFIG_OVERRIDE_JSON={config_override_json}",
        "-e", "PYTHONPATH=/app",
        # Volumes
        "-v", f"{host_client_root}:/app/client-data",
        "-v", f"{host_global_root}:/app/global-artifacts",
        "-v", f"{host_data}:/app/data:ro",
        "-v", f"{host_config}:/app/config:ro",
        # Stdout/stderr written inside container → captured via Popen
        image,
        "python", "-m", "client.runtime",
        "--client-id", client_id,
        "--server-address", server_address,
        "--artifact-root", "/app/client-data",
        "--max-retries", "30",
        "--retry-delay", "3",
    ]
    return cmd


def run_local_training(
    user_context: dict[str, Any], run_config: dict[str, Any]
) -> dict[str, Any]:
    artifact_root = str(user_context["artifact_root"])
    metadata_path = build_artifact_paths(artifact_root)["metadata"]
    if not metadata_path.exists():
        raise RuntimeError(
            f"Phase 2 preprocessing has not been run for this run. "
            f"Expected metadata at: {metadata_path}. "
            f"Run Phase 2 (preprocessing) before starting training."
        )

    tracer = PipelineTracer(artifact_root, run_id=user_context.get("run_id", ""))

    progress_callback = user_context.get("progress_callback")

    tracer.mark("prepare_data", "start")
    if progress_callback is not None:
        progress_callback("prepare_data", "Preparing local data split...")
    prepare_data(user_context)
    tracer.mark("prepare_data", "end")

    global_artifact_root = str(user_context.get("global_artifact_root", artifact_root))
    timeout_seconds = int(user_context.get("client_timeout_seconds", 600))

    deployment_mode = os.environ.get("DEPLOYMENT_MODE", "subprocess")

    external_server = user_context.get("server_address")
    if external_server:
        server_address = external_server
    elif deployment_mode == "docker_per_user":
        portal_host = os.environ.get("PORTAL_HOSTNAME", socket.gethostname())
        server_address = f"{portal_host}:{find_free_port()}"
    else:
        server_address = f"127.0.0.1:{find_free_port()}"
    spawned_server = external_server is None
    pythonpath = str(Path(__file__).resolve().parents[1])

    override_for_run = dict(user_context.get("config_override") or {})
    override_for_run.setdefault("artifacts", {})
    override_for_run["artifacts"].update(
        {
            "root": global_artifact_root,
            "models_dir": str(Path(global_artifact_root) / "models"),
            "metrics_dir": str(Path(global_artifact_root) / "metrics"),
            "xai_dir": str(Path(global_artifact_root) / "xai"),
            "audit_dir": str(Path(global_artifact_root) / "audit"),
            "reports_dir": str(Path(global_artifact_root) / "reports"),
        }
    )

    log_dir = Path(artifact_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    server_process: subprocess.Popen[str] | None = None
    if spawned_server:
        server_stdout = log_dir / "server_stdout.log"
        server_stderr = log_dir / "server_stderr.log"

        server_env = {
            **dict(os.environ),
            "PYTHONPATH": pythonpath,
            "FL_SERVER_ADDRESS": server_address,
            "FL_FORCE_EXIT_ON_COMPLETE": "1",
            "CONFIG_OVERRIDE_JSON": json.dumps(override_for_run),
        }

        tracer.mark("server_launch", "start", detail=f"address={server_address}")
        _emit_progress_from_traces(
            artifact_root=artifact_root,
            global_artifact_root=global_artifact_root,
            progress_callback=progress_callback,
        )
        with (
            server_stdout.open("w", encoding="utf-8") as out_fh,
            server_stderr.open("w", encoding="utf-8") as err_fh,
        ):
            server_process = subprocess.Popen(
                [sys.executable, "-m", "server.run_training"],
                env=server_env,
                stdout=out_fh,
                stderr=err_fh,
            )
        tracer.mark("server_launch", "end", detail=f"pid={server_process.pid}")
    else:
        tracer.mark("server_reuse", "start", detail=f"address={server_address}")
        tracer.mark("server_reuse", "end")

    # ── Resolve effective num_clients ─────────────────────────────────────────
    num_clients = int(
        (user_context.get("config_override") or {})
        .get("federated_learning", {})
        .get("num_clients", 1)
    )
    if num_clients < 1:
        num_clients = 1

    # ── Prepare per-client artifact directories when num_clients > 1 ─────────
    # When the portal is configured for multiple clients, we partition the
    # training data from artifact_root into N disjoint splits and create N
    # lightweight client directories that share the same val/test/metadata but
    # each hold only their own training partition.  The first client is
    # identified by the logged-in user; additional clients get synthetic IDs.
    client_roots: list[str] = []
    client_ids: list[str] = []
    base_paths = build_artifact_paths(artifact_root)

    if num_clients > 1:
        from shared.federated_data import partition_dataset_for_clients

        fl_clients_dir = Path(artifact_root) / "fl_clients"
        client_splits_dirs: list[Path] = []
        primary_client_id = str(user_context["client_id"])

        for i in range(num_clients):
            cid = primary_client_id if i == 0 else f"client_{i + 1}"
            client_ids.append(cid)
            croot = fl_clients_dir / cid
            splits_dir = croot / "data" / "splits"
            meta_dir = croot / "data" / "metadata"
            splits_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)
            # Copy shared artifacts: metadata, validation, test
            (meta_dir / "preprocessing_metadata.json").write_bytes(
                base_paths["metadata"].read_bytes()
            )
            (splits_dir / "validation.npz").write_bytes(
                base_paths["validation"].read_bytes()
            )
            (splits_dir / "test.npz").write_bytes(
                base_paths["test"].read_bytes()
            )
            # Copy effective config
            effective_cfg = Path(artifact_root) / "effective_config.json"
            if effective_cfg.exists():
                (croot / "effective_config.json").write_bytes(
                    effective_cfg.read_bytes()
                )
            client_roots.append(str(croot))
            client_splits_dirs.append(splits_dir)

        # Partition training data across clients (stratified, disjoint)
        try:
            partition_dataset_for_clients(
                train_path=base_paths["train"],
                client_dirs=client_splits_dirs,
                seed=42,
            )
        except ValueError as _pe:
            # Not enough samples to partition — fall back to single client
            _logger.warning(
                "Cannot partition train data for %d clients (%s). "
                "Falling back to single-client training.",
                num_clients,
                _pe,
            )
            num_clients = 1
            client_roots = []
            client_ids = []

    if not client_ids:
        # Single client path (original behaviour)
        client_roots = [artifact_root]
        client_ids = [str(user_context["client_id"])]

    # ── Build base client env ─────────────────────────────────────────────────
    base_client_env = {
        **dict(os.environ),
        "PYTHONPATH": pythonpath,
        "GLOBAL_ARTIFACT_ROOT": global_artifact_root,
        "CONFIG_OVERRIDE_JSON": json.dumps(override_for_run),
    }

    # One log file per client
    client_procs: list[subprocess.Popen[str]] = []
    client_log_paths: list[tuple[Path, Path]] = []

    try:
        tracer.mark("client_training", "start")
        _emit_progress_from_traces(
            artifact_root=artifact_root,
            global_artifact_root=global_artifact_root,
            progress_callback=progress_callback,
        )

        if server_process is not None:
            time.sleep(1.0)
            if server_process.poll() is not None:
                stderr_log = log_dir / "server_stderr.log"
                stderr_tail = (
                    stderr_log.read_text(encoding="utf-8")[-2000:]
                    if stderr_log.exists()
                    else ""
                )
                raise RuntimeError(
                    f"FL server exited immediately after launch "
                    f"(rc={server_process.returncode}): {stderr_tail}"
                )

        for idx, (cid, croot) in enumerate(zip(client_ids, client_roots)):
            c_stdout_path = log_dir / f"client_{idx}_stdout.log"
            c_stderr_path = log_dir / f"client_{idx}_stderr.log"
            client_log_paths.append((c_stdout_path, c_stderr_path))

            if deployment_mode == "docker_per_user":
                client_cmd = _build_docker_client_cmd(
                    client_id=cid,
                    server_address=server_address,
                    artifact_root=croot,
                    global_artifact_root=global_artifact_root,
                    config_override_json=json.dumps(override_for_run),
                    log_dir=log_dir,
                )
                with (
                    c_stdout_path.open("w", encoding="utf-8") as c_out,
                    c_stderr_path.open("w", encoding="utf-8") as c_err,
                ):
                    client_procs.append(subprocess.Popen(
                        client_cmd,
                        stdout=c_out,
                        stderr=c_err,
                        text=True,
                    ))
            else:
                per_client_env = {
                    **base_client_env,
                    "CLIENT_ARTIFACT_ROOT": croot,
                    "CLIENT_CONFIG_PATH": str(Path(croot) / "effective_config.json"),
                }
                with (
                    c_stdout_path.open("w", encoding="utf-8") as c_out,
                    c_stderr_path.open("w", encoding="utf-8") as c_err,
                ):
                    client_procs.append(subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "client.runtime",
                            "--client-id", cid,
                            "--server-address", server_address,
                            "--artifact-root", croot,
                            "--max-retries", "20",
                            "--retry-delay", "1",
                        ],
                        env=per_client_env,
                        stdout=c_out,
                        stderr=c_err,
                        text=True,
                    ))
            _logger.info("Spawned client %s (root=%s)", cid, croot)
            # Small stagger to avoid simultaneous gRPC connection storms
            if idx > 0:
                time.sleep(0.5)

        deadline = time.monotonic() + timeout_seconds
        while any(proc.poll() is None for proc in client_procs):
            _emit_progress_from_traces(
                artifact_root=artifact_root,
                global_artifact_root=global_artifact_root,
                progress_callback=progress_callback,
            )
            if server_process is not None:
                src = server_process.poll()
                if src is not None and src != 0:
                    for proc in client_procs:
                        proc.kill()
                    stderr_log = log_dir / "server_stderr.log"
                    stderr_tail = (
                        stderr_log.read_text(encoding="utf-8")[-2000:]
                        if stderr_log.exists()
                        else ""
                    )
                    raise RuntimeError(
                        f"FL server died during training (rc={src}): {stderr_tail}"
                    )
            if time.monotonic() >= deadline:
                for proc in client_procs:
                    proc.kill()
                raise RuntimeError(
                    f"FL clients timed out after {timeout_seconds}s "
                    f"(clients: {client_ids})."
                )
            time.sleep(2.0)

        # Check all client return codes
        for idx, (proc, cid, (c_stdout_path, c_stderr_path)) in enumerate(
            zip(client_procs, client_ids, client_log_paths)
        ):
            client_rc = proc.returncode
            if client_rc != 0:
                client_err_tail = (
                    c_stderr_path.read_text(encoding="utf-8")[-2000:]
                    if c_stderr_path.exists()
                    else ""
                )
                client_out_tail = (
                    c_stdout_path.read_text(encoding="utf-8")[-2000:]
                    if c_stdout_path.exists()
                    else ""
                )
                raise RuntimeError(
                    f"FL client runtime failed for {cid}: "
                    f"{client_err_tail or client_out_tail}"
                )

        tracer.mark("client_training", "end", detail=f"clients={len(client_procs)}")
        _emit_progress_from_traces(
            artifact_root=artifact_root,
            global_artifact_root=global_artifact_root,
            progress_callback=progress_callback,
        )

        # Backward-compat aliases so existing error-display code still works
        client_stdout_path = client_log_paths[0][0] if client_log_paths else log_dir / "client_stdout.log"
        client_stderr_path = client_log_paths[0][1] if client_log_paths else log_dir / "client_stderr.log"
        # keep legacy names in log_dir for UI log display
        if not (log_dir / "client_stdout.log").exists() and client_log_paths:
            try:
                (log_dir / "client_stdout.log").write_bytes(client_log_paths[0][0].read_bytes())
                (log_dir / "client_stderr.log").write_bytes(client_log_paths[0][1].read_bytes())
            except Exception:
                pass

        if server_process is not None:
            tracer.mark("server_shutdown", "start")
            deadline_shutdown = time.monotonic() + 120.0
            server_return: int | None = None
            while time.monotonic() < deadline_shutdown:
                _emit_progress_from_traces(
                    artifact_root=artifact_root,
                    global_artifact_root=global_artifact_root,
                    progress_callback=progress_callback,
                )
                server_return = server_process.poll()
                if server_return is not None:
                    break
                time.sleep(1.0)
            if server_return is None:
                server_process.kill()
                tracer.mark("server_shutdown", "end", detail="killed_after_timeout")
                raise RuntimeError(
                    "FL server did not shut down after training completed."
                )
            tracer.mark("server_shutdown", "end", detail=f"returncode={server_return}")
            if server_return != 0:
                stderr_log = log_dir / "server_stderr.log"
                stderr_tail = (
                    stderr_log.read_text(encoding="utf-8")[-2000:]
                    if stderr_log.exists()
                    else ""
                )
                raise RuntimeError(f"FL server subprocess failed: {stderr_tail}")
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.kill()
        for proc in client_procs:
            if proc.poll() is None:
                proc.kill()
        tracer.flush()

    metrics_path = build_artifact_paths(global_artifact_root)["metrics"]
    if not metrics_path.exists():
        raise RuntimeError(
            "Federated training completed without producing run metrics."
        )
    _logger.info("Pipeline timing summary: %s", tracer.summary())
    return json.loads(metrics_path.read_text(encoding="utf-8"))
