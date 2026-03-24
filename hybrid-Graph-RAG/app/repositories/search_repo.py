from app.repositories.neo4j_client import Neo4jClient


def _build_filter_clause(
    date_from: str | None,
    date_to: str | None,
    rooms: list[str],
    users: list[str],
) -> tuple[str, dict]:
    conditions = []
    parameters: dict[str, object] = {}

    if date_from:
        conditions.append("m.date >= $date_from")
        parameters["date_from"] = date_from
    if date_to:
        conditions.append("m.date <= $date_to")
        parameters["date_to"] = date_to
    if rooms:
        conditions.append("r.name IN $rooms")
        parameters["rooms"] = rooms
    if users:
        conditions.append("u.name IN $users")
        parameters["users"] = users

    clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return clause, parameters


def _append_condition(clause: str, condition: str) -> str:
    if not clause:
        return f"WHERE {condition}"
    return f"{clause} AND {condition}"


class SearchRepository:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def fetch_vector_candidates(
        self,
        query_vector: list[float],
        date_from: str | None,
        date_to: str | None,
        rooms: list[str],
        users: list[str],
        limit: int,
    ) -> list[dict]:
        clause, parameters = _build_filter_clause(date_from, date_to, rooms, users)
        clause = _append_condition(clause, "m.embedding_status = 'completed'")
        parameters.update({"query_vector": query_vector, "limit": limit})
        return self.client.run_read(
            f"""
            CALL db.index.vector.queryNodes("message_embedding_index", $limit, $query_vector)
            YIELD node, score
            WITH node AS m, score
            MATCH (u:User)-[:SENT]->(m)-[:IN_ROOM]->(r:Room)
            {clause}
            RETURN
              DISTINCT
              m.message_id AS message_id,
              m.occurred_at AS occurred_at,
              m.date AS date,
              m.time AS time,
              r.name AS room_name,
              u.name AS user_name,
              m.content AS content,
              score AS score
            ORDER BY score DESC
            """,
            parameters,
        )

    def search_fulltext(
        self,
        query: str,
        date_from: str | None,
        date_to: str | None,
        rooms: list[str],
        users: list[str],
        limit: int,
    ) -> list[dict]:
        clause, parameters = _build_filter_clause(date_from, date_to, rooms, users)
        parameters.update({"query": query, "limit": max(limit, 20)})
        return self.client.run_read(
            f"""
            CALL db.index.fulltext.queryNodes("message_content_fulltext", $query)
            YIELD node, score
            WITH node AS m, score
            MATCH (u:User)-[:SENT]->(m)-[:IN_ROOM]->(r:Room)
            {clause}
            RETURN
              DISTINCT
              m.message_id AS message_id,
              m.occurred_at AS occurred_at,
              m.date AS date,
              m.time AS time,
              r.name AS room_name,
              u.name AS user_name,
              m.content AS content,
              score AS score
            ORDER BY score DESC
            LIMIT $limit
            """,
            parameters,
        )

    def fetch_message_by_id(self, message_id: str) -> dict | None:
        rows = self.client.run_read(
            """
            MATCH (u:User)-[:SENT]->(m:Message {message_id: $message_id})-[:IN_ROOM]->(r:Room)
            RETURN
              m.message_id AS message_id,
              m.occurred_at AS occurred_at,
              m.date AS date,
              m.time AS time,
              r.name AS room_name,
              u.name AS user_name,
              m.content AS content
            LIMIT 1
            """,
            {"message_id": message_id},
        )
        return rows[0] if rows else None

    def fetch_context(self, message_id: str) -> dict[str, list[dict]]:
        previous_in_room = self.client.run_read(
            """
            MATCH (m:Message {message_id: $message_id})-[:PREV_IN_ROOM*1..2]->(prev:Message)
            MATCH (u:User)-[:SENT]->(prev)-[:IN_ROOM]->(r:Room)
            RETURN
              prev.message_id AS message_id,
              prev.occurred_at AS occurred_at,
              prev.date AS date,
              prev.time AS time,
              r.name AS room_name,
              u.name AS user_name,
              prev.content AS content
            ORDER BY prev.occurred_at DESC
            """,
            {"message_id": message_id},
        )
        next_in_room = self.client.run_read(
            """
            MATCH (next:Message)-[:PREV_IN_ROOM*1..2]->(m:Message {message_id: $message_id})
            MATCH (u:User)-[:SENT]->(next)-[:IN_ROOM]->(r:Room)
            RETURN
              next.message_id AS message_id,
              next.occurred_at AS occurred_at,
              next.date AS date,
              next.time AS time,
              r.name AS room_name,
              u.name AS user_name,
              next.content AS content
            ORDER BY next.occurred_at ASC
            """,
            {"message_id": message_id},
        )
        recent_by_user = self.client.run_read(
            """
            MATCH (u:User)-[:SENT]->(m:Message {message_id: $message_id})
            MATCH (u)-[:SENT]->(other:Message)-[:IN_ROOM]->(r:Room)
            WHERE other.message_id <> m.message_id
            RETURN
              other.message_id AS message_id,
              other.occurred_at AS occurred_at,
              other.date AS date,
              other.time AS time,
              r.name AS room_name,
              u.name AS user_name,
              other.content AS content
            ORDER BY other.occurred_at DESC
            LIMIT 3
            """,
            {"message_id": message_id},
        )
        same_day_same_room_samples = self.client.run_read(
            """
            MATCH (u:User)-[:SENT]->(m:Message {message_id: $message_id})-[:IN_ROOM]->(r:Room)
            MATCH (other_u:User)-[:SENT]->(other:Message)-[:IN_ROOM]->(r)
            WHERE other.date = m.date AND other.message_id <> m.message_id
            RETURN
              other.message_id AS message_id,
              other.occurred_at AS occurred_at,
              other.date AS date,
              other.time AS time,
              r.name AS room_name,
              other_u.name AS user_name,
              other.content AS content
            ORDER BY other.occurred_at ASC
            LIMIT 5
            """,
            {"message_id": message_id},
        )
        return {
            "previous_in_room": previous_in_room,
            "next_in_room": next_in_room,
            "recent_by_user": recent_by_user,
            "same_day_same_room_samples": same_day_same_room_samples,
        }
