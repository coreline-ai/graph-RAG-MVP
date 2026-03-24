CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.document_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT user_name_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.name IS UNIQUE;

CREATE CONSTRAINT channel_name_unique IF NOT EXISTS
FOR (c:Channel) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT date_value_unique IF NOT EXISTS
FOR (d:Date) REQUIRE d.date IS UNIQUE;
