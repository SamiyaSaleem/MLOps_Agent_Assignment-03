import json
import re
import subprocess


# Restrict all operations to the assignment's local cluster.
CLUSTER_CONTEXT = "kind-mlops-agent"
NAMESPACE = "mlops-agent"

DEPLOYMENT_NAME = "mlops-inference"
SERVICE_NAME = "mlops-inference-service"

POD_PATTERN = re.compile(
    r"^mlops-inference-[a-z0-9]+-[a-z0-9]+$"
)


def run_kubectl(arguments, timeout=20):
    """
    Execute a bounded kubectl command against the approved
    local Kind cluster and namespace.
    """

    command = [
        "kubectl",
        "--context",
        CLUSTER_CONTEXT,
        "--namespace",
        NAMESPACE,
        *arguments,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Kubernetes command timed out",
        }

    except FileNotFoundError:
        return {
            "status": "error",
            "error": "kubectl executable is unavailable",
        }

    if result.returncode != 0:
        return {
            "status": "error",
            "error": "Kubernetes command failed",
            "exit_code": result.returncode,
        }

    return {
        "status": "success",
        "output": result.stdout,
    }


def get_json_resource(arguments):
    """Retrieve and parse JSON from Kubernetes."""

    result = run_kubectl([*arguments, "-o", "json"])

    if result["status"] != "success":
        return result

    try:
        data = json.loads(result["output"])

    except (json.JSONDecodeError, TypeError):
        return {
            "status": "error",
            "error": "Invalid Kubernetes JSON response",
        }

    return {
        "status": "success",
        "data": data,
    }


def list_kubernetes_resources():
    """List approved namespace deployments, Pods and Services."""

    resources = {}

    for resource_type in (
        "deployments",
        "pods",
        "services",
    ):
        result = get_json_resource(
            ["get", resource_type]
        )

        if result["status"] != "success":
            return result

        resources[resource_type] = [
            {
                "name": item.get(
                    "metadata", {}
                ).get("name"),
                "phase": item.get(
                    "status", {}
                ).get("phase"),
            }
            for item in result["data"].get("items", [])
        ]

    return {
        "status": "success",
        "cluster": CLUSTER_CONTEXT,
        "namespace": NAMESPACE,
        "resources": resources,
    }


def get_deployment_health():
    """Check desired, available and ready inference replicas."""

    result = get_json_resource(
        ["get", "deployment", DEPLOYMENT_NAME]
    )

    if result["status"] != "success":
        return result

    deployment = result["data"]

    spec = deployment.get("spec", {})
    status = deployment.get("status", {})

    desired = spec.get("replicas", 1)
    ready = status.get("readyReplicas", 0)
    available = status.get("availableReplicas", 0)

    return {
        "status": "success",
        "deployment": DEPLOYMENT_NAME,
        "namespace": NAMESPACE,
        "desired_replicas": desired,
        "ready_replicas": ready,
        "available_replicas": available,
        "healthy": (
            desired > 0
            and ready == desired
            and available == desired
        ),
    }


def get_service_endpoints():
    """Retrieve actual inference Service endpoint addresses."""

    result = get_json_resource([
        "get",
        "endpointslices",
        "-l",
        f"kubernetes.io/service-name={SERVICE_NAME}",
    ])

    if result["status"] != "success":
        return result

    endpoints = []

    for item in result["data"].get("items", []):
        for endpoint in item.get("endpoints", []):
            endpoints.append({
                "addresses": endpoint.get("addresses", []),
                "ready": endpoint.get(
                    "conditions", {}
                ).get("ready"),
            })

    return {
        "status": "success",
        "service": SERVICE_NAME,
        "namespace": NAMESPACE,
        "endpoints": endpoints,
        "endpoint_count": len(endpoints),
    }


def get_pod_logs(pod_name, tail_lines=50):
    """
    Retrieve a bounded diagnostic summary of an approved Pod.

    Raw logs are deliberately not returned to the AI agent.
    """

    if (
        not isinstance(pod_name, str)
        or not POD_PATTERN.fullmatch(pod_name)
    ):
        return {
            "status": "error",
            "error": "Unapproved Pod name",
        }

    if (
        type(tail_lines) is not int
        or not 1 <= tail_lines <= 100
    ):
        return {
            "status": "error",
            "error": "Invalid log line limit",
        }

    #Verify the Pod belongs to this specific inference app
    pod_result = get_json_resource(
        ["get", "pod", pod_name]
    )

    if pod_result["status"] != "success":
        return pod_result

    labels = pod_result["data"].get(
        "metadata", {}
    ).get("labels", {})

    if labels.get("app") != DEPLOYMENT_NAME:
        return {
            "status": "error",
            "error": "Pod does not belong to approved application",
        }

    result = run_kubectl([
        "logs",
        pod_name,
        "-c",
        "inference-api",
        "--tail",
        str(tail_lines),
    ])

    if result["status"] != "success":
        return result

    log_text = result["output"][:20000]

    patterns = {
        "model_loading_failure": (
            "model loading failed",
            "filenotfounderror",
        ),
        "application_error": (
            "traceback",
            "exception",
        ),
        "resource_failure": (
            "outofmemory",
            "memoryerror",
        ),
    }

    detected = []

    for category, keywords in patterns.items():
        if any(
            keyword in log_text.lower()
            for keyword in keywords
        ):
            detected.append(category)

    return {
        "status": "success",
        "pod_name": pod_name,
        "diagnostic_categories": detected,
        "lines_requested": tail_lines,
        "note": (
            "Actual Pod logs inspected. "
            "Raw logs withheld to avoid exposing sensitive data."
        ),
    }


def scale_inference_api(
    replicas,
    confirmed=False,
):
    """
    Scale the approved inference deployment.

    This is a state-changing operation.
    """

    if type(replicas) is not int:
        return {
            "status": "error",
            "error": "Replica count must be an integer",
        }

    if not 1 <= replicas <= 3:
        return {
            "status": "error",
            "error": "Allowed replica count is 1 to 3",
        }

    if confirmed is not True:
        return {
            "status": "confirmation_required",
            "requested_replicas": replicas,
            "deployment": DEPLOYMENT_NAME,
            "message": (
                "Explicit user approval is required "
                "before changing Kubernetes replicas."
            ),
        }

    result = run_kubectl([
        "scale",
        "deployment",
        DEPLOYMENT_NAME,
        f"--replicas={replicas}",
    ])

    if result["status"] != "success":
        return result

    return {
        "status": "success",
        "deployment": DEPLOYMENT_NAME,
        "requested_replicas": replicas,
        "message": "Scaling command accepted by Kubernetes",
    }
