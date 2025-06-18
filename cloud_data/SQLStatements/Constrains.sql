ALTER TABLE "games" ADD CONSTRAINT "user_in_game" FOREIGN KEY ("user_id") REFERENCES "game_user" ("id");
ALTER TABLE "game_data" ADD CONSTRAINT "game_id_in_game_data" FOREIGN KEY ("game_id") REFERENCES "games" ("game_id");
ALTER TABLE "game_data" ADD CONSTRAINT "user_input_in_game_data" FOREIGN KEY ("user_input") REFERENCES "game_inputs" ("id");