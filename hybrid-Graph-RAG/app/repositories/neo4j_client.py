from collections.abc import Callable

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self, raise_on_error: bool = True) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Neo4jError:
            if raise_on_error:
                raise
            return False

    def run_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write(self, query: str, parameters: dict | None = None) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(lambda tx: list(tx.run(query, parameters or {})))
            return [record.data() for record in result]

    def execute_write(self, work: Callable, *args, **kwargs):
        with self.driver.session(database=self.database) as session:
            return session.execute_write(work, *args, **kwargs)

    def execute_read(self, work: Callable, *args, **kwargs):
        with self.driver.session(database=self.database) as session:
            return session.execute_read(work, *args, **kwargs)

    def fetch_total_messages(self) -> int:
        rows = self.run_read("MATCH (m:Message) RETURN count(m) AS count")
        return int(rows[0]["count"]) if rows else 0

    def fetch_last_ingestion_timestamp(self) -> str | None:
        rows = self.run_read(
            """
            MATCH (m:Message)
            RETURN max(m.occurred_at) AS last_ingestion_timestamp
            """
        )
        if not rows:
            return None
        return rows[0]["last_ingestion_timestamp"]
