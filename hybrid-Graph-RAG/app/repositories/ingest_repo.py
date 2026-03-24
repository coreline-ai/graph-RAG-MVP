from dataclasses import asdict

from app.models.records import ChatMessageRecord
from app.repositories.neo4j_client import Neo4jClient


class IngestRepository:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def upsert_messages(self, records: list[ChatMessageRecord]) -> None:
        if not records:
            return
        rows = [asdict(record) for record in records]
        self.client.run_write(
            """
            UNWIND $rows AS row
            MERGE (m:Message {message_id: row.message_id})
            SET m.occurred_at = row.occurred_at,
                m.date = row.date,
                m.time = row.time,
                m.content = row.content,
                m.source_file = row.source_file,
                m.line_no = row.line_no,
                m.embedding_status = row.embedding_status,
                m.embedding = row.embedding
            MERGE (u:User {name: row.user_name})
            MERGE (r:Room {name: row.room_name})
            MERGE (d:Date {date: row.date})
            MERGE (u)-[:SENT]->(m)
            MERGE (m)-[:IN_ROOM]->(r)
            MERGE (m)-[:ON_DATE]->(d)
            """,
            {"rows": rows},
        )

    def clear_prev_in_room_relationships(self) -> None:
        self.client.run_write("MATCH (:Message)-[rel:PREV_IN_ROOM]->(:Message) DELETE rel")

    def create_prev_in_room_relationships(self) -> int:
        rows = self.client.run_write(
            """
            MATCH (r:Room)<-[:IN_ROOM]-(m:Message)
            WITH r, m
            ORDER BY r.name, m.occurred_at, m.line_no
            WITH r, collect(m) AS messages
            UNWIND CASE WHEN size(messages) < 2 THEN [] ELSE range(1, size(messages) - 1) END AS idx
            WITH messages[idx] AS current, messages[idx - 1] AS previous
            MERGE (current)-[:PREV_IN_ROOM]->(previous)
            RETURN count(*) AS created
            """
        )
        return int(rows[0]["created"]) if rows else 0

    def fetch_counts(self) -> dict[str, int]:
        rows = self.client.run_read(
            """
            CALL {
              MATCH (m:Message)
              RETURN count(m) AS messages
            }
            CALL {
              MATCH (u:User)
              RETURN count(u) AS users
            }
            CALL {
              MATCH (r:Room)
              RETURN count(r) AS rooms
            }
            CALL {
              MATCH (d:Date)
              RETURN count(d) AS dates
            }
            CALL {
              MATCH ()-[rel:PREV_IN_ROOM]->()
              RETURN count(rel) AS prev_links
            }
            RETURN messages, users, rooms, dates, prev_links
            """
        )
        if not rows:
            return {"messages": 0, "users": 0, "rooms": 0, "dates": 0, "prev_links": 0}
        row = rows[0]
        return {
            "messages": int(row["messages"]),
            "users": int(row["users"]),
            "rooms": int(row["rooms"]),
            "dates": int(row["dates"]),
            "prev_links": int(row["prev_links"]),
        }
