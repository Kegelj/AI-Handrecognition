SELECT * FROM games
SELECT * FROM game_data
SELECT * FROM game_user
SELECT * FROM game_inputs
	
SELECT * FROM staging_gamedata
SELECT * FROM staging_bbox
Drop table game_user


INSERT INTO game_user (name) SELECT 'Peter'
WHERE NOT EXISTS (
    SELECT 1 FROM game_user WHERE name = 'Peter'
);

INSERT INTO game_user (name) VALUES ('Peter')
INSERT INTO game_user (name) VALUES ('Peter'),('Herbert') ON CONFLICT DO NOTHING;
insert into staging_gamedata (game_id,user_name,user_input,timestamp) VALUES ('45BX05','Peter',5,12.15)

UPDATE staging_gamedata set processed=False;

select distinct game_id from staging_gamedata where processed=False
	
SELECT * FROM staging_gamedata WHERE processed=False
SELECT DISTINCT user_name from staging_gamedata WHERE processed = False

ALTER TABLE staging_gamedata
ADD COLUMN player_health INTEGER,
ADD COLUMN player_x INTEGER,
ADD COLUMN player_y INTEGER,
ADD COLUMN goat_kills INTEGER,
ADD COLUMN squirrel_kills INTEGER;
