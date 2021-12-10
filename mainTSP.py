#Traveling salesman problem
#Ernesto Guzmán Saleh @ernestosaleh

from GeneticTravelModel import *
#Import Data
data15=importCoordinates("15Cities.txt")
data48=importCoordinates("48Cities.txt")

popSize=50 #Pupulation Size
iters=10**3 #Iterations

#Choose your model between:
    #GeneticModel01
    #GeneticModel02
    #GeneticModel03
#Look at docstrings for parameter examples or at the jupyter notebook TSP.ipynb
histPOP, histDist, pop=GeneticModel01(*data15, popSize, iters, 1, True)
#histPOP, histDist, pop=GeneticModel02(*data15, popSize, iters, 10, 2)
#histPOP, histDist, pop=GeneticModel03(*data15, popSize, iters, 2, True, 2, 5)


#Plot distance evolution and trayectory
print("Distance", histDist[-1])
plotDistance(histDist)
plotTrayectory(*data15, histPOP[-1])