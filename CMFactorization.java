package com.qqx.cmf;

import Jama.Matrix;

public class CMFactorization {
	
	
	public static void main(String[] args) {
		Matrix B = new Matrix(new double[][]{{1.0,2,3}, {4,5,6}, {7,8,9}});
		Matrix bVec = B.getMatrix(1, 1, 0, 2);
		bVec.print(1, 4);
	}
	
	
	public Matrix cmf(Matrix Y, int maxIter, int K, double lambdaD, double lambdaT, double lambdaL){
		int nd = Y.getRowDimension();
		int nt = Y.getColumnDimension();
		Matrix resMat = new Matrix(nd, nt);
		
		//初始化变量
		Matrix wd = Matrix.random(1, nd);
		Matrix wt = Matrix.random(1, nt);
		Matrix A = Matrix.random(nd, K).times(0.1);
		Matrix B = Matrix.random(nt, K).times(0.1);
		
		for(int r=0; r<maxIter; r++){
			Matrix simD = A.times(A.transpose());
			Matrix simT = B.times(B.transpose());
			
			//更新矩阵A中的元素
			for(int i=0; i<nd; i++){
				Matrix topLeftVec = new Matrix(1, K);
				for(int j=0; j<nt; j++){
					double wij = Y.get(i, j);
					wij = Math.pow(wij, 2);
					
					Matrix bVec = B.getMatrix(j, j, 0, K-1);
					bVec.timesEquals(wij);
					topLeftVec.plusEquals(bVec);	// topLeftVec = topLeftVec.plus(bVec);
				}
				
				Matrix topRightVec = new Matrix(1, K);
				for(int p=0; p<nd; p++){
					double w = 0;
					for(int k=0; k<nd; k++){
						w += wd.get(0, k) * simD.get(i, p);
					}
					Matrix aVec = A.getMatrix(p, p, 0, K-1);
					aVec.timesEquals(w);
					topRightVec.plusEquals(aVec);
				}
				topRightVec.timesEquals(lambdaD);
				
				Matrix bottonLeftMat = new Matrix(K, K);
				for(int j=0; j<nt; j++){
					double wij = Y.get(i, j);
					Matrix bVec = B.getMatrix(j, j, 0, K-1);
					bottonLeftMat.plusEquals(bVec.transpose().times(bVec).times(wij));
				}
				
				Matrix bottonRighttMat = new Matrix(K, K);
				for(int j=0; j<nd; j++){
					Matrix aVec = A.getMatrix(j, j, 0, K-1);
					bottonRighttMat.plusEquals(aVec.transpose().times(aVec));
				}
				bottonRighttMat.timesEquals(lambdaD);
				
				//根据公式合并各个部分
				Matrix botton = bottonLeftMat.plus(bottonRighttMat).plus(Matrix.identity(K, K).times(lambdaL));
				botton = botton.inverse();
				Matrix top = topLeftVec.plus(topRightVec);
				A.setMatrix(i, i, 0, K-1, top.times(botton));
			}
			
			//更新矩阵B中的元素
			for(int j=0; j<nt; j++){
				Matrix topLeftVec = new Matrix(1, K);
				for(int i=0; i<nd; i++){
					double wij = Y.get(i, j);
					wij = Math.pow(wij, 2);
					
					Matrix aVec = A.getMatrix(i, i, 0, K-1);
					aVec.timesEquals(wij);
					topLeftVec.plusEquals(aVec);
				}
				
				Matrix topRightVec = new Matrix(1, K);
				for(int q=0; q<nt; q++){
					double w = 0;
					for(int k=0; k<nt; k++){
						w += wt.get(0, k) * simT.get(j, q);
					}
					Matrix bVec = B.getMatrix(q, q, 0, K-1);
					bVec.timesEquals(w);
					topRightVec.plusEquals(bVec);
				}
				topRightVec.timesEquals(lambdaT);
				
				Matrix bottonLeftMat = new Matrix(K, K);
				for(int i=0; i<nd; i++){
					double wij = Y.get(i, j);
					Matrix aVec = A.getMatrix(i, i, 0, K-1);
					bottonLeftMat.plusEquals(aVec.transpose().times(aVec).times(wij));
				}
				
				Matrix bottonRighttMat = new Matrix(K, K);
				for(int q=0; q<nt; q++){
					Matrix bVec = B.getMatrix(q, q, 0, K-1);
					bottonRighttMat.plusEquals(bVec.transpose().times(bVec));
				}
				bottonRighttMat.timesEquals(lambdaT);
				
				//根据公式合并各个部分
				Matrix botton = bottonLeftMat.plus(bottonRighttMat).plus(Matrix.identity(K, K).times(lambdaL));
				botton = botton.inverse();
				Matrix top = topLeftVec.plus(topRightVec);
				B.setMatrix(j, j, 0, K-1, top.times(botton));
			}
			
		}
		
		return resMat;
	}

}
