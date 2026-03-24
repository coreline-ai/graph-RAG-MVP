from app.repositories.neo4j_client import Neo4jClient


SCHEMA_STATEMENTS = [
    """
    CREATE CONSTRAINT message_id_unique IF NOT EXISTS
    FOR (m:Message) REQUIRE m.message_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT user_name_unique IF NOT EXISTS
    FOR (u:User) REQUIRE u.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT room_name_unique IF NOT EXISTS
    FOR (r:Room) REQUIRE r.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT date_unique IF NOT EXISTS
    FOR (d:Date) REQUIRE d.date IS UNIQUE
    """,
    """
    CREATE VECTOR INDEX message_embedding_index IF NOT EXISTS
    FOR (m:Message) ON (m.embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
      }
    }
    """,
    """
    CREATE FULLTEXT INDEX message_content_fulltext IF NOT EXISTS
    FOR (m:Message) ON EACH [m.content]
    """,
    """
    CREATE INDEX message_date_idx IF NOT EXISTS
    FOR (m:Message) ON (m.date)
    """,
    """
    CREATE INDEX message_occurred_at_idx IF NOT EXISTS
    FOR (m:Message) ON (m.occurred_at)
    """,
]


def ensure_schema(client: Neo4jClient) -> None:
    for statement in SCHEMA_STATEMENTS:
        client.run_write(statement)
