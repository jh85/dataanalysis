import numpy as np

def minor_submatrix(m,r,c):
    mask = np.ones_like(m, dtype=bool)
    mask[r,:] = False
    mask[:,c] = False
    dim = len(m)
    return m[mask].reshape(dim-1,dim-1)

def mydet(m):
    if len(m) == 1:
        return m[0,0]
    
    total = 0
    for i in range(len(m)):
        total += (-1)**i * m[0,i] * mydet(minor_submatrix(m,0,i))
    return total

def main():
    dim = 8
    m = np.random.normal(size=dim*dim).reshape(dim,dim)
    print(np.linalg.det(m))
    print(mydet(m))

main()
