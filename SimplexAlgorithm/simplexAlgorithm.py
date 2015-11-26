import numpy as np

class Simplex():
    def __init__(self, simplexTableau):
        """
        Given a matrix which represents a maximization problem in standard form,
        initialize the Simplex object.
        """
        self.simplexTableau = simplexTableau
        self.numberOfVariables = len(simplexTableau[0])
        self.invertedSimplex =  simplexTableau.T 
        self.rhsValues = self.invertedSimplex[-1]
        self.bigM = 10**10 ### Should be changed if you're actually dealing with numbers like this

    def findEnteringVariable(self):
        """
        Find the entering variable by finding the minimum coefficient in the
        objective function and return its index.
        """
        pivotColumn = self.simplexTableau[0].argmin()
        print "Pivot Column:", pivotColumn
        return pivotColumn

    def ratioTest(self, pivotColumn):
        """
        Perform the ratio test with the given column index number and return
        the row index around which we will pivot.
        """
        ratioList = [float(x[0])/x[1] if x[1] > 0 else self.bigM for x in zip(self.rhsValues, self.invertedSimplex[pivotColumn])]
        pivotRow = ratioList.index(min(ratioList))
        print "Ratio List:", ratioList
        print "Pivot Row:", pivotRow
        return pivotRow

    def makeBasicVariable(self, pivotRow, pivotColumn):
        """
        Given the pivoting column and row, perform row operations on the simplex
        tableau so that the entering variable is a basic variable.
        """
        a = self.simplexTableau[pivotRow][pivotColumn]
        pivotingRow = [(1.0/a) * i for i in self.simplexTableau[pivotRow]]
        self.simplexTableau = np.delete(self.simplexTableau,pivotRow,axis = 0)
        newSimplexTableau = list()
        for row in self.simplexTableau:
                a = row[pivotColumn] * -1
                newRow = [i[0] + i[1]*a for i in zip(row, pivotingRow)]
                newSimplexTableau.append(newRow)
        newSimplexTableau.insert(pivotRow, pivotingRow)
        self.simplexTableau = np.array(newSimplexTableau)
        self.invertedSimplex = [[row[i] for row in self.simplexTableau] for i in range(self.numberOfVariables)]
        self.rhsValues = self.invertedSimplex[-1]
        print self.simplexTableau
        
    def checkOptimal(self):
        if min(self.simplexTableau[0]) >= 0:
                return True
        else:
                return False

    def solve(self):
        print "Initial Tableau\n" + "-" * 20
        print self.simplexTableau
        print "\n\n"

        while not self.checkOptimal():
            column = self.findEnteringVariable()
            row = self.ratioTest(column)
            self.makeBasicVariable(row, column)
            print "\n\n"

if __name__ == "__main__":

    tableau = [[1, -3, -2, 0, 0, 0, 0],
               [0, 2, 1, 1, 0, 0, 100],
               [0, 1, 1, 0, 1, 0, 80],
               [0, 1, 0, 0, 0, 1, 40]]

    a = Simplex(np.array(tableau))
    a.solve()
    print "Another Cases"
    tableau = [[1, -60, -35, -20, 0, 0, 0, 0, 0],
               [0, 8, 6, 1, 1, 0, 0, 0, 48],
               [0, 4, 2, 1.5, 0, 1, 0, 0, 20],
               [0, 2, 1.5, 0.5, 0, 0, 1, 0, 8],
               [0, 0, 1, 0, 0, 0, 0, 1, 5]]

    b = Simplex(np.array(tableau))
    b.solve()
