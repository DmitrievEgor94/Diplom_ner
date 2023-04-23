CREATE TABLE IF NOT EXISTS violations (
   row_id INTEGER PRIMARY KEY AUTOINCREMENT,
   date DATE NOT NULL,
   text text NOT NULL,
   name text NOT NULL,
   bad_words NOT NULL
);
