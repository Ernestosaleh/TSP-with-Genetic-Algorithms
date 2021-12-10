from typing import ChainMap
import numpy as np
from numpy.lib.function_base import average
from matplotlib import pyplot as plt
from numpy import math

#----------------------------------------------------------------------
#Evaluation Functions 
#----------------------------------------------------------------------

def importCoordinates(file):
    """
    Imports coordinates x, y, s from tab separated columns txt file into lists
        Params:
            file: Obligatory columns are (x, y), Optional is s
        Outputs:
            x, y (1D np.array): Coordinates
            s (float): average standard deviation if provided, default is s=1.
    """
    with open(file, "r") as f:
        data=f.readlines()
    #The elements are separated b tab (\t) in a
    x=np.array([]); y=np.array([]); s=np.array([])
    for line in data: #Removal of headers
        line=line.replace("\n","") #Removal of (\n)
        lineinlist=line.split(" ") #Converts str into list
        while "" in lineinlist:
            lineinlist.remove("")
        fvals=[float(x) for x in lineinlist] #String to float
        x=np.append(x, [fvals[0]]); 
        y=np.append(y, [fvals[1]])
    return(x,y)

def coordinateMatrix(x, y):
    """
    Converts coordinates to a Distance Matrix (DM) of NxN.
    e.g.: Distance from a0 to a2 is DM[0][2] while ai to aj is DM[i][j] same as DM[j][i].
        Params:
            x, y (1D np.array): Coordinates
        Output:
            DM (2D np.array): Positive square symetric Distance Matrix.
    """
    n=np.size(x)
    DM=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j: #Trivial case
                DM[i][j]=0
            elif j<i: #Avoids double calculattions
                DM[i][j]=DM[j][i]
            else:
                DM[i][j]=np.sqrt((x[j]-x[i])**2+(y[j]-y[i])**2)
    return(DM)


#----------------------------------------------------------------------
#Genetic Functions
#----------------------------------------------------------------------

def genPopulation(DM, popSize, quitDups=True, backHome=True):
    """
    Creation of Pupulation.
        Params:
            DM (np.array): Distance Matrix
            popSize (int): Size of the population.
            quitDups (bool): If False, admits repeated individuals. 
                If True, PopSize (if greater) reduces to (N-1)!. 
                N is the number of cities.
            backHome (bool): If True, the first destination is also de final one.
        Output   
            pop (np.array): 2D matrix where each line is an individual.
    Note: 
        Each individual represents a list of ordered travel destinations.
        e.g. indv=[0, 3, 5, 2, 8, 1,...,0] with size N+1 if backHome
        Each individual is randomized with np.random.shuffle function
    """
    N=np.size(DM[0]) #Number of cities
    #Ordered ascendent vector [0, 1, 2,..., N-1]
    OGvec=np.arange(N)
    pop=[] 
    if quitDups and popSize>=math.factorial(N-1):
        popSize=math.factorial(N-1)
    for i in range(popSize):
        success=False
        while not success:
            success=True
            np.random.shuffle(OGvec[1:]) #Random shuffle
            indv=np.append(OGvec[0], OGvec[1:]) #An individual
            if backHome:
                indv=np.append(indv, OGvec[0])
            if quitDups:
                for testindv in pop:
                    if np.array_equal(testindv,indv):
                        success=False
        pop.append(indv)
    pop=np.array(pop)
    return(pop)

#Total distance traveled in an individual
totDistance=lambda indv, DM: np.sum([DM[indv[i],indv[i+1]] for i in range(len(indv)-1)])  

def aptitudes(pop, DM):
    """
    Aptitude of an individual: 1/Traveled distance
        Params:
            pop (np.array): selected Population
            DM (np.array): Distance Matrix
        Output:
            apts (np.array): popoulation aptitudes.
    Note: 
        Each individual represents a list of ordered travel destinations.
    """
    distances=np.array([totDistance(indv, DM) for indv in pop])
    apts=1/distances
    return(apts, distances) 

   
def selection(apts, N=2):
    """
    Selection op couples. Propability an individual getting selected is proportional to its aptitude.
        Params:
            apts (np.array): popoulation aptitudes.
        Output:
            coupleIDs (np.array): IDs of a couple of individuals in the population.
    Note: 
        Each individual represents a list of ordered travel destinations.
        I wish this couple many healthy children and a happy life.
    """
    sumAPT=np.sum(apts)
    cumAPT=np.cumsum(apts)  ##Cumulative Aptitude Probability
    coupleIDs=[]
    while len(coupleIDs)!=N: #Gets two non identycal indexes.
        rn=np.random.uniform(0, sumAPT) ##Randon number
        indvID=np.where(cumAPT>=rn)[0][0] ##Find the closest number above rn
        if indvID not in coupleIDs:
            coupleIDs.append(indvID)
    coupleIDs=np.array(coupleIDs)
    return(coupleIDs)


def proCreate00(pop, coupleIDS, doublekids=False):
    """
    This function generates 2 offsprings by a displacement of values in one individual
    biased by a second one. (Individuals are vectors)
        Params:
            pop (np.array): selected Population
            coupleIDS (np.array): 2 indexes of individuals in a population.
        Output:
            children (np.array): Offsprings of the couple.
    Notes: 
    Parents are divided in halves. 
    Vector 1 replaces its first half with the first half of vector 2.
    Vector 1 displaces its original values to second half, where the new ones of the first half were before.
    That was the first kid.
    Second kid is the same but now the main vector is vector 2.
    Third and Fourth kids follows the same idea but now  the last half of each parent goes first.
    """ 
    N=np.size(pop[coupleIDS[0]])
    halfN=int(N/2)
    if doublekids: #A factor multiplies the original number of kids
        mult=2  
    else:
        mult=1
    children=np.zeros((mult*2,N))  
    #Parents are obtained from original data
    parents=np.array([pop[coupleIDS[0]], pop[coupleIDS[1]]])
    #Stars process for mult pairs of kids
    for k in range(mult):
        if k==1:
            #If doubleKids then the second pair of parents is created from original data
            prep1=np.append(pop[coupleIDS[0]][halfN:-1], [pop[coupleIDS[0]][1:halfN]])
            prep2=np.append(pop[coupleIDS[1]][halfN:-1], [pop[coupleIDS[1]][1:halfN]])
            parent1=np.append(np.zeros(1), np.append(prep1, [0]))
            parent2=np.append(np.zeros(1), np.append(prep2, [0]))
            parents=np.array([parent1, parent2])
        #Starts the procreation process
        for j in range(2):
            if j==1:
                #For the second child, parents are reversed.
                parents=np.array([parents[1], parents[0]])
            children[j+2*k][0]=parents[0][0]
            children[j+2*k]=parents[1]
            from_idx=1
            prechild=np.copy(parents[1])
            for i in parents[0][1:halfN]:
                to_idx=np.where(prechild==i)[0][0]
                children[j+2*k][from_idx]=parents[0][from_idx]
                children[j+2*k][to_idx]=prechild[from_idx]
                prechild=np.copy(children[j+2*k])
                from_idx+=1
    return(children)

def mutation01(children, criteria=0.1, backHome=True):
    """
    Children are exposed to radiaton. Oh no! Will they mutate?
    If a child does, a random pair of its coefficientes change order.
        Params:
            children(np.array): Offsprings of the couple.
            criteria(float): Between [0,1]. Represents the probability of mutation
            backHome: Considers that the traveler goes back home.
        Output:
            children(np.array): Childrens after radiation exposition.         
    """
    for idchild in range(len(children)):
        child=children[idchild]
        newchild=np.copy(child)
        rn=np.random.uniform(0,1)
        if backHome:
            uplim=len(child)-1
        else:
            uplim=len(child)
        if rn<=criteria:
            if backHome:
                rid1=np.random.randint(1,uplim)
            else:
                rid1=np.random.randint(1,uplim)
            rid2=rid1
            while rid2==rid1: #Non repeated indexes
                rid2=np.random.randint(1, uplim)
            newchild[rid1]=child[rid2]
            newchild[rid2]=child[rid1]
        children[idchild]=newchild
    return(children)

def cloneMutation(pop, popIDS=np.array([]), Nmuts=10):
    """
    Takes individuals IDs, clones them and mutates them Nmuts times by shuffling 2 random places
        Params:
            pop(np.array): Pupulation with different individuals.
            popIDS(np.array): IDs of the individuals to by cloned. 
                If null then all pop is cloned.
            Nmuts(int): Number of mutations on a single individual
        Output:
            Children(np.array): Cloned and mutated individuals.
    Note:
        Is this ethical? Maybe not.
    """
    
    if np.size(popIDS)>0:
        children=pop[popIDS] #Clone certain individuals
    else:
        children=np.copy(pop) #Clone all population
    for _ in range(Nmuts):
        for cID in np.arange(0, len(children[:,0])):   
            rind=np.random.randint(1,len(children[0])-1, 2)
            children[cID,rind[0]], children[cID,rind[1]]=children[cID,rind[1]],children[cID,rind[0]]  
    return children


def filterPopulation(pop, children, oldAPT, oldDIST, DM):
    """
    Filters the expanded population according to its aptitud so it can go back to its original size.
        Params:
            x(np.array): x coordinate array.
            y(np.array): y coordinate array.
            pop(np.array): Pupulation with different individuals.
            oldAPT(np.array): Aptitude of the original sized population.
            oldDIST(np.array): Traveled Distances of the original sized population.
        Output:
            pop(np.array): New pupulation with original size. Individuals with less aptitude removed. 
            popAPT(np.array): Aptitudes of the new population.
            popDIST(np.array): Traveled Distance of the new population.
            topINDV(np.array): The fittest Individual.
            minDIST(float): The traveled distance of the fittest Individual.
            DM (np.array): Distance Matrix
    """
    childrenAPT, childDIST=aptitudes(children, DM)
    for i in range(len(children)):
        pop=np.vstack((pop, children[i]))
    popAPT=np.append(oldAPT, childrenAPT)
    popDIST=np.append(oldDIST, childDIST)
    ascendIDS=np.argsort(popAPT)
    #Number of low IDs is the size of the new childrens. 
    #So we can take off the exact amount later.
    low_ids=ascendIDS[0:len(children)] 
    max_id=ascendIDS[-1] #top individual
    topINDV=pop[max_id]
    minDIST=popDIST[max_id]
    popAPT=np.delete(popAPT, low_ids, 0)
    pop=np.delete(pop, low_ids, 0)
    popDIST=np.delete(popDIST, low_ids, 0)
    return(pop, popAPT, popDIST, topINDV, minDIST)

#----------------------------------------------------------------------
#Genetic Full Models
#----------------------------------------------------------------------
def GeneticModel01(x,y, popSize, iters, P, dbkids=False):
    """
    Creates offsprings according to proCreate00 function with a probability P of mutation.
        Params:
            x(np.array): x coordinate array.
            y(np.array): y coordinate array.
            P(float): Probability of a kid to mutate with function mutation01.
            dbkids(bool): if True, a couple breeds 4 kids instead of 2 in function proCreate00.
        Outputs:
            histPOP(np.array): History of fittest individuals per iteration.
            histDist(np.array): History of minimal distances per iteration.
            pop(np.array): Population from last generation.
    """
    #History of top individuals with min distances.
    histPOP=[]; histDist=[]
    #Distances matrix
    DM=coordinateMatrix(x,y)
    #Pupulation Creation Section
    pop=genPopulation(DM, popSize) #population
    apts, dist=aptitudes(pop, DM) #aptitudes
    for _ in range(iters):
        #Population selection for crossover
        coupleIDS=selection(apts)
        #Children generation
        children=proCreate00(pop, coupleIDS, dbkids)
        #Mutations
        children=mutation01(children, P)
        children=children.astype(int)

        #Filter
        (pop, popAPT, popDIST, topINDV, minDIST)=filterPopulation(pop, children, apts, dist, DM)

        #Preparations for next iteration.
        apts=np.copy(popAPT); dist=np.copy(popDIST)
        histPOP.append(topINDV); histDist.append(minDIST)
    return histPOP, histDist, pop

def GeneticModel02(x,y, popSize, iters, clonekids, mutsPerKid):
    """
    Evolution by clonation and mutation. The fittest individuals create clonekids with mutsPerKid.
        Params:
            x(np.array): x coordinate array.
            y(np.array): y coordinate array.
            popSize(int): Size of the population.
            iters(int): number of iterations.
            mutsPerKid(int): number of mutations by function CloneMutation.
            clonekids(int): Number of kids done by only clonation and mutation from a parent.
        Outputs:
            histPOP(np.array): History of fittest individuals per iteration.
            histDist(np.array): History of minimal distances per iteration.
            pop(np.array): Population from last generation.
    """
    #History of top individuals with min distances.
    histPOP=[]; histDist=[]
    #Distances matrix
    DM=coordinateMatrix(x,y)
    #Pupulation Creation Section
    pop=genPopulation(DM, popSize) #population
    apts, dist=aptitudes(pop, DM) #aptitudes
    for _ in range(iters):
        
        #Population selection for crossover
        popIDS=selection(apts, clonekids)

        #Solo reproduction with mutation.
        children=cloneMutation(pop, popIDS, mutsPerKid)
        children=children.astype(int)
        #Filter
        (pop, popAPT, popDIST, topINDV, minDIST)=filterPopulation(pop, children, apts, dist, DM)
    
        #Preparations for next iteration.
        apts=np.copy(popAPT); dist=np.copy(popDIST)
        histPOP.append(topINDV); histDist.append(minDIST)
    return histPOP, histDist, pop

def GeneticModel03(x,y, popSize, iters, mutsPerKid, dbkids=False, natCouples=1 ,clonekids=0):
    """
    This is an hybrid model from Model01 and Model02.
        Params:
            x(np.array): x coordinate array.
            y(np.array): y coordinate array.
            popSize(int): Size of the population.
            iters(int): number of iterations.
            mutsPerKid(int): number of mutations by function CloneMutation.
            dbkids(bool): if True, a couple breeds 4 kids instead of 2 in function proCreate00.
            natCouples(int): Number of couples that spawn the kids in proCreate.
            clonekids(int): Number of kids done by only clonation and mutation from a parent.
        Outputs:
            histPOP(np.array): History of fittest individuals per iteration.
            histDist(np.array): History of minimal distances per iteration.
            pop(np.array): Population from last generation.
    """
    #History of top individuals with min distances.
    histPOP=[]; histDist=[]
    #Distances matrix
    DM=coordinateMatrix(x,y)
    Ncities=np.size(x)

    #Pupulation Creation Section
    pop=genPopulation(DM, popSize) #population
    apts, dist=aptitudes(pop, DM) #aptitudes

    #Number of individuals to select.
    if clonekids>2:
        select=clonekids
    else:
        select=2
    if natCouples+1>select:
        select=natCouples+1
    for _ in range(iters):

        #Population selection for crossover
        popIDS=selection(apts, select)

        #Children generation
        children=np.zeros(Ncities+1)
        #By natural parents
        for j in range(natCouples):
            newchildren=proCreate00(pop, popIDS[j:j+2], dbkids)
            #newchildren=mutation01(newchildren, 0.8)
            newchildren=cloneMutation(newchildren, Nmuts=mutsPerKid) #Mutation
            children=np.vstack((children, newchildren))

        #By conlation.
        if clonekids>0:
            newchildren=cloneMutation(pop, popIDS[0:clonekids], Nmuts=mutsPerKid) #Mutation
            children=np.vstack((children, newchildren))

        children=np.delete(children, 0, 0)
        children=children.astype(int)
        
        #Filter population
        (pop, popAPT, popDIST, topINDV, minDIST)=filterPopulation(pop, children, apts, dist, DM)
    
        #Preparations for next iteration.
        apts=np.copy(popAPT); dist=np.copy(popDIST)
        histPOP.append(topINDV); histDist.append(minDIST)
    return histPOP, histDist, pop

#----------------------------------------------------------------------
#Graphers
#----------------------------------------------------------------------

def plotDistance(histDIST):
    """Plots the Distance of the fittest individual per iteration"""
    plt.rcParams["font.family"] = "serif"
    plt.title("Distance Evolution"); plt.xlabel("Generation"); plt.ylabel("Distance(u.l.)")
    plt.plot(histDIST, linewidth=3, color="blue")
    plt.grid()
    plt.show()

def plotTrayectory(x,y,topINDV):
    """Plots the trayectory of the traveling salesman"""
    plt.rcParams["font.family"] = "serif"
    xpath=x[topINDV]; ypath=y[topINDV]
    plt.plot(xpath,ypath)  
    plt.title("Map of cities and trayectory travelled by the fittest individual.")
    plt.xlabel("Distance(u.l.)"); plt.ylabel("Distance(u.l.)") 
    plt.plot(x,y,'r.',markersize=20, label="Cities")
    plt.plot(x[0], y[0], 'g.', markersize=20, label="Initial city")
    plt.legend()
    plt.show()

def plotDM(DM):
    """
    Does a color map of the distance matrix
    """
    plt.title("Distances Matrix.")
    plt.xlabel("Cities"); plt.ylabel("Cities") 
    color_map = plt.imshow(DM)
    color_map.set_cmap("Blues")
    plt.rcParams["font.family"] = "serif"
    plt.colorbar(label="Distance(u.l)")
    plt.legend()
    plt.show()
