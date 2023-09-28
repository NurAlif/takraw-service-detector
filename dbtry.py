import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="dbku"
)

print(mydb)

def read_all_matches():
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM matches")
    myresult = mycursor.fetchall()

    return myresult

def read_replays(match_id):
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM replays where match_id="+str(match_id))
    myresult = mycursor.fetchall()

    return myresult

# create_new_replay(2)
# create_new_replay(2)

print(read_replays(2))