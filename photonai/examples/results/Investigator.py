
from photonai.investigator.Investigator import Investigator

# from mongodb
Investigator.load_from_db('mongodb://server:27017/photon_results', 'my_pipe')

# from working memory
my_pipe = Hyperpipe(...)
my_pipe.fit(X, y)
Investigator.show(my_pipe)

# from file
Investigator.load_from_file('my_pipe_name', '/home/usr123/proj45/my_pipe_results2019-08-01/photon_result_file.p')
