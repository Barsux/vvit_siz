import psycopg2
# Демо
from time import time, sleep

AVG_FOR_FRAME = 0.913
OBJECTS_TOTAL = 214
STARTTIME = time()
# print(STARTTIME)
sleep(2)
ENDTIME = time()


# Демо кончается

class Data_db:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="172.28.1.3",
            database="siz_db",
            user="chugun",
            password="123456",
            port=5432,
            connect_timeout=5)

        self.cursor = self.conn.cursor()
        self.table_exists()

    def __del__(self):
        pass

    def table_exists(self):
        self.cursor.execute("select exists(select * from information_schema.tables where table_name=%s)",
                            ('website_analysis',))
        check_row = self.cursor.fetchone()[0]

        if not check_row:
            self.cursor.execute('''create table website_analysis(id_check int GENERATED ALWAYS AS IDENTITY,
                                                            start_time numeric null,
                                                            end_time numeric null,
                                                            average_frame_processing_time text null,
                                                            number_of_detected_objects int null)''')

    def insert_data(self, starttime, endtime, avg_for_frame, objects_total):
        SQL_Insert = f"INSERT INTO website_analysis (start_time, end_time, average_frame_processing_time, " \
                     f"number_of_detected_objects) values ({starttime}, {endtime}, {avg_for_frame}, {objects_total}); "
        self.cursor.execute(SQL_Insert)
        self.conn.commit()
        self.conn.close()


if __name__ == "__main__":
    DB = Data_db()
    DB.insert_data(STARTTIME, ENDTIME, AVG_FOR_FRAME, OBJECTS_TOTAL)
