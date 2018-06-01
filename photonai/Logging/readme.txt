Note: The logger currently uses a separate database (photon_log_db).
This should not be the case. We need one single database that contains
all of the programs collections -> Set your database 



0) You may need to change the lines

    client = MongoClient('localhost', 27017)
    log_db = client.your_db

to match your database


1) Import the logger with

      from Logging import Logger

2) Start logging. Example:

       Logger.info('Your log message')


-------------------------------


- Debug should be used for information that may be useful for program-	debugging
- Info should be used if something interesting (but uncritically) happened, 
  like the results of a computation
- Something may have gone wrong? Use warning!
- Something broke down. Error should be used if something unexpected happened!

-------------------------------


You can change the LogLevel by using
	Logger.set_log_level(LogLevel.WARN)
	(this should be done at the beginning in most cases -
	Test/Development: Info or Debug,
	Production: Warn)

-------------------------------

If you want to use the BadLoggerGUI, you also have to change
the used collections to match your own collection

    Trigger.subscribe_to_collection('photon_manager_db.debug_log', add_log, Operation.INSERT)


(We need to find one db name for all groups!)