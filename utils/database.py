import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="dbku"
)

print(mydb)



def create_new_match(match_name):
    mycursor = mydb.cursor()

    sql = "INSERT INTO matches (name, created_at) VALUES ('"+match_name+"', NOW())"
    mycursor.execute(sql)
    mycursor.lastrowid
    mydb.commit()

    return mycursor.lastrowid

def create_new_replay(match_id, time_str, foul, recording_str):
    mycursor = mydb.cursor()

    sql = "INSERT INTO replays (time_service, recording, foul, match_id) VALUES ('"+time_str+"', '"+recording_str+"', "+str(foul)+", "+str(match_id)+")"
    mycursor.execute(sql)
    mycursor.lastrowid
    mydb.commit()

    return mycursor.lastrowid

def read_all_matches():
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM matches")
    myresult = mycursor.fetchall()

    return myresult

def read_match_name(id):
    mycursor = mydb.cursor()

    mycursor.execute("SELECT name FROM matches where id="+str(id))
    myresult = mycursor.fetchone()[0]

    return myresult

def read_replays(match_id):
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM replays where match_id="+str(match_id))
    myresult = mycursor.fetchall()

    return myresult

def read_recordpath(replay_id):
    mycursor = mydb.cursor()

    mycursor.execute("SELECT recording FROM replays where id="+str(replay_id))
    myresult = mycursor.fetchone()

    return myresult[0]