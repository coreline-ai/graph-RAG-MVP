from fastapi import Request


def require_neo4j_connection(request: Request) -> None:
    neo4j_client = getattr(request.app.state, "neo4j_client", None)
    if neo4j_client is None:
        raise RuntimeError("neo4j client is not available")
    if not neo4j_client.verify_connectivity(raise_on_error=False):
        raise RuntimeError("neo4j is not connected")


def require_search_service(request: Request):
    require_neo4j_connection(request)
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise RuntimeError("search service is not available")
    return service


def require_insights_service(request: Request):
    require_neo4j_connection(request)
    service = getattr(request.app.state, "insights_service", None)
    if service is None:
        raise RuntimeError("insights service is not available")
    return service
