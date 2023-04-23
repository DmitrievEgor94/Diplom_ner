def create_table_if_not_exists(sql_con):
    with open('create_table_script.sql') as f:
        sql_script = f.read()

    sql_con.execute(sql_script)