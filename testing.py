import pickle

with open('memory.pickle', 'rb') as f: 
    memory = pickle.load(f)

print ('loaded_obj is', memory.size())
