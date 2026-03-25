import logging
from truefoundry.deploy import (
    Build,
    Resources,
    NodeSelector,
    PythonBuild,
    LocalSource,
    Port,
    Service,
)

logging.basicConfig(level=logging.INFO)

service = Service(
    name="prompt-tune-backend-v0",
    image=Build(
        build_source=LocalSource(),
        build_spec=PythonBuild(
            build_context_path=".",
            command="uvicorn src.chat.controller.chat_controller:app --host 0.0.0.0 --port 21120",
        ),
    ),
    resources=Resources(
        cpu_request=0.01,
        cpu_limit=1.0,
        memory_request=1000,
        memory_limit=1000,
        ephemeral_storage_request=500,
        ephemeral_storage_limit=500,
        node=NodeSelector(capacity_type="spot_fallback_on_demand"),
    ),
    env={
        "TFY_API_KEY": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImxzV0lDNWtkU1V1bXg1ckg5NkR6bFdYUGxJTSJ9.eyJhdWQiOiI4OTUyNTNhZi1lYzlkLTRiZTYtODNkMS02ZjI0OGU2NDRlNzkiLCJleHAiOjM3MjE0OTIzNjgsImlhdCI6MTc2MTk0MDM2OCwiaXNzIjoidHJ1ZWZvdW5kcnkuY29tIiwic3ViIjoiY21oZjlzd3R3MDc3MzAxcmdkazdpNDkxdyIsImp0aSI6ImNtaGY5c3d1MDA3NzQwMXJnMjA4YjVvbWUiLCJzdWJqZWN0U2x1ZyI6ImRlZmF1bHQtY21naG9mYXBxMDJsNDAxc2M4YWxtYXIyYiIsInVzZXJuYW1lIjoiZGVmYXVsdC1jbWdob2ZhcHEwMmw0MDFzYzhhbG1hcjJiIiwidXNlclR5cGUiOiJzZXJ2aWNlYWNjb3VudCIsInN1YmplY3RUeXBlIjoic2VydmljZWFjY291bnQiLCJ0ZW5hbnROYW1lIjoidHJ1ZWZvdW5kcnkiLCJyb2xlcyI6W10sImp3dElkIjoiY21oZjlzd3UwMDc3NDAxcmcyMDhiNW9tZSIsImFwcGxpY2F0aW9uSWQiOiI4OTUyNTNhZi1lYzlkLTRiZTYtODNkMS02ZjI0OGU2NDRlNzkifQ.NaYRS65FHgD9XxN7_w43pKcWChErmGallENuUGHMg_8HmUuwITS5Upr_qq3si0K_eEnuDo5kPH_9PzdFFXuFe9X7JzBX_8xFw__88_3ALT6BpUXI3pOKn5GnEixOQSlbpvP9d7tbyYF38Mmv87_2DfcZ9C5v6daiZV-H5db7RqnGdEkTOF4r-BVRo5eJ7O3R0DYTwL_s8sQ7s5AyK2-Y-nY2admluyVebglUexsMy9b9SuD9BygXtawEQ8uwG2e1jrpYYb0BQ_kPdVV4DGQGcmHizf-mAIbQ7P5f6ZlPke2Esp9talr4SJoDQagC0k351PAup6L0XIm9PA3U5MC6mA",
        "TFY_HOST": "https://internal.devtest.truefoundry.tech/",
        "LLM_BASE_URL": "https://tfy-llm-gateway-test-truefoundry-8787.tfy-usea1-ctl.devtest.truefoundry.tech",
        "CONFIGBASEPATH": "config",
    },
    ports=[
        Port(
            port=21120,
            protocol="TCP",
            expose=True,
            app_protocol="http",
            host="prompt-tune-backend-v0-harsh-ws-21120.tfy-usea1-ctl.devtest.truefoundry.tech",
        )
    ],
    replicas=1.0,
)


service.deploy(workspace_fqn="tfy-usea1-devtest:harsh-ws", wait=False)
