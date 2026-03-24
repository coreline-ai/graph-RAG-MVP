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


class InsightsRepository:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def messages_by_date(
        self, date_from: str | None, date_to: str | None, rooms: list[str], users: list[str]
    ) -> list[dict]:
        clause, parameters = _build_filter_clause(date_from, date_to, rooms, users)
        return self.client.run_read(
            f"""
            MATCH (u:User)-[:SENT]->(m:Message)-[:IN_ROOM]->(r:Room)
            {clause}
            RETURN m.date AS key, count(*) AS count
            ORDER BY key ASC
            """,
            parameters,
        )

    def top_rooms(
        self,
        date_from: str | None,
        date_to: str | None,
        rooms: list[str],
        users: list[str],
        limit: int = 10,
    ) -> list[dict]:
        clause, parameters = _build_filter_clause(date_from, date_to, rooms, users)
        parameters["limit"] = limit
        return self.client.run_read(
            f"""
            MATCH (u:User)-[:SENT]->(m:Message)-[:IN_ROOM]->(r:Room)
            {clause}
            RETURN r.name AS key, count(*) AS count
            ORDER BY count DESC, key ASC
            LIMIT $limit
            """,
            parameters,
        )

    def top_users(
        self,
        date_from: str | None,
        date_to: str | None,
        rooms: list[str],
        users: list[str],
        limit: int = 10,
    ) -> list[dict]:
        clause, parameters = _build_filter_clause(date_from, date_to, rooms, users)
        parameters["limit"] = limit
        return self.client.run_read(
            f"""
            MATCH (u:User)-[:SENT]->(m:Message)-[:IN_ROOM]->(r:Room)
            {clause}
            RETURN u.name AS key, count(*) AS count
            ORDER BY count DESC, key ASC
            LIMIT $limit
            """,
            parameters,
        )

    def keyword_samples(
        self,
        date_from: str | None,
        date_to: str | None,
        rooms: list[str],
        users: list[str],
        keywords: list[str],
    ) -> list[dict]:
        results: list[dict] = []
        for keyword in keywords:
            clause, parameters = _build_filter_clause(date_from, date_to, rooms, users)
            parameters["keyword"] = keyword
            clause = _append_condition(clause, "m.content CONTAINS $keyword")
            rows = self.client.run_read(
                f"""
                MATCH (u:User)-[:SENT]->(m:Message)-[:IN_ROOM]->(r:Room)
                {clause}
                RETURN
                  $keyword AS keyword,
                  m.message_id AS message_id,
                  m.occurred_at AS occurred_at,
                  r.name AS room_name,
                  u.name AS user_name,
                  m.content AS content
                ORDER BY m.occurred_at DESC
                LIMIT 1
                """,
                parameters,
            )
            if rows:
                results.append(rows[0])
        return results
