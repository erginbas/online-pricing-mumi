def replace_submatrix(mat, ind1, ind2, mat_replace):
  for i, index in enumerate(ind1):
    mat[index, ind2] = mat_replace[i, :]
  return mat