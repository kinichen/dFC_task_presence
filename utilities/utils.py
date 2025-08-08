import math
import numpy as np



def get_n_ROI(a, b, c): # solves quadratic in ax**2+bx+c=0 form.
	discriminant = b**2 - 4*a*c
	if discriminant < 0:
		return ValueError("No real roots")
	root1 = (-b + math.sqrt(discriminant)) / (2*a)  # always returns a float
	root2 = (-b - math.sqrt(discriminant)) / (2*a)
	
	if root1 > 0:
		if root1.is_integer():
			return int(root1)
		else:
			return ValueError(f"Number of ROIs = {root1} is not an integer")
	else:
		if root2.is_integer():
			return int(root2)
		else:
			return ValueError(f"Number of ROIs = {root2} is not an integer")


def vec_to_symmetric_matrix(vec, roi):
	mat = np.zeros((roi, roi))
	idx = np.triu_indices(roi, k=1)	# excludes diagonal
	mat[idx] = vec
	mat = mat + mat.T
	return mat