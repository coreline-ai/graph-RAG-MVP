from app.repositories.neo4j_client import Neo4jClient
from app.repositories.schema import ensure_schema
from app.settings import get_settings


def main() -> None:
    settings = get_settings()
    client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        ensure_schema(client)
        print("Neo4j schema is ready.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
