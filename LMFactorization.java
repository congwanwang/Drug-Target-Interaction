package com.qqx.cmf;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import com.qqx.common.FileUtils;

import Jama.Matrix;

public class LMFactorization {
	
	public static void main(String[] args) {
		File file = new File("C:\\Users\\HP\\Desktop\\new 3.txt");
		Matrix simMat = readSimMatrix(file);
		simMat.print(1, 3);
	}
	
	public static Matrix readSimMatrix(File file){
		List<String> lines = FileUtils.readFileByLine(file);
		int num = lines.get(0).split("	").length-1;
		Matrix simMat = new Matrix(num, num);
		for(int i=1; i<lines.size(); i++){
			String[] elems = lines.get(i).split("	");
			if(elems.length == 0){
				continue;
			}
			for(int j=1; j<=num; j++){
				simMat.set(i-1, j-1, Double.parseDouble(elems[j]));
			}
		}
		return simMat;
	}
	
	public Matrix lmf(Matrix Y, Matrix simD, Matrix simT, int maxIter, int r, int c, int K, double lambdaD,
			double lambdaT, double alpha, double beta, double gamma){
		int row = Y.getRowDimension();
		int col = Y.getColumnDimension();
		Matrix resMat = new Matrix(row, col);
		
		//初始化变量
		Matrix U = Matrix.random(row, r).times(0.1);
		Matrix V = Matrix.random(col, r).times(0.1);
		Matrix A = initialSimMat(simD, K);
//		System.out.println("A  "+simD.getRowDimension()+", "+simD.getColumnDimension());
		Matrix ldMat = getLMatrix(A);
//		System.out.println(ldMat.getRowDimension()+", "+ldMat.getColumnDimension());
		
		Matrix B = initialSimMat(simT, K);
		Matrix ltMat = getLMatrix(B);
		
		Matrix uPhi = new Matrix(row, r);
		Matrix lPhi = new Matrix(col, r);
		for(int iter=0; iter<maxIter; iter++){
			Matrix P = new Matrix(row, col);
			resMat = U.times(V.transpose());
			for(int i=0; i<row; i++){
				for (int j = 0; j < col; j++) {
					double pij = resMat.get(i, j);
					P.set(i, j, Math.pow(Math.E, pij)/(1+Math.pow(Math.E, pij)));
				}
			}
			
			//计算U、V的梯度
			Matrix pv = P.times(V);
			Matrix ypv = Y.arrayTimes(P).times(V).times(c-1);
			Matrix yv = Y.times(V).times(c);
//			System.out.println(U.getRowDimension()+", "+U.getColumnDimension());
			Matrix ilu = ldMat.times(alpha).plus(Matrix.identity(row, row).times(lambdaD)).times(U);
			Matrix gdMat = pv.plus(ypv).minus(yv).plus(ilu);
			
			Matrix pu = P.transpose().times(U);
			Matrix ypu = Y.transpose().arrayTimes(P.transpose()).times(U).times(c-1);
			Matrix yu = Y.transpose().times(U).times(c);
			Matrix ilv = ltMat.times(beta).plus(Matrix.identity(col, col).times(lambdaT)).times(V);
			Matrix gtMat = pu.plus(ypu).minus(yu).plus(ilv);
			
			for(int i=0; i<row; i++){
				for(int k=0; k<r; k++){
					double gik = gdMat.get(i, k);
					double phiik = uPhi.get(i, k)+Math.pow(gik, 2);
					uPhi.set(i, k, phiik);
					double uik = U.get(i, k);
					U.set(i, k, uik-gamma*gik/Math.pow(phiik, 0.5));
				}
			}
			
			for(int j=0; j<col; j++){
				for(int k=0; k<r; k++){
					double gjk = gtMat.get(j, k);
					double phijk = lPhi.get(j, k)+Math.pow(gjk, 2);
					lPhi.set(j, k, phijk);
					double vjk = V.get(j, k);
					V.set(j, k, vjk-gamma*gjk/Math.pow(phijk, 0.5));
				}
			}
		}
		resMat = U.times(V.transpose());
		
		return resMat;
	}
	
	public Matrix getLMatrix(Matrix simMat){
		int row = simMat.getRowDimension();
		int col = simMat.getColumnDimension();
		Matrix resMat = new Matrix(row, col);		
		Matrix diMat = new Matrix(row, col);
		for(int i=0; i<row; i++){
			double sum = 0;
			for (int j = 0; j < col; j++) {
				sum += simMat.get(i, j);
			}
			diMat.set(i, i, sum);
		}
		
		Matrix djMat = new Matrix(row, col);
		for (int j = 0; j < col; j++) {
			double sum = 0;
			for(int i=0; i<row; i++){
				sum += simMat.get(i, j);
			}
			djMat.set(j, j, sum);
		}
		
		resMat = diMat.plus(djMat).minus(simMat.plus(simMat.transpose()));
		return resMat;
	}
	
	public Matrix initialSimMat(Matrix simMat, int K){
		int row = simMat.getRowDimension();
		int col = simMat.getColumnDimension();
		Matrix resMat = new Matrix(row, col);
		
		for(int i=0; i<row; i++){
			List<SortInfomation> list = new ArrayList<>();
			for (int j = 0; j < col; j++) {
				list.add(new SortInfomation(simMat.get(i, j), i, j));
			}

			Collections.sort(list, new Comparator<SortInfomation>() {
				@Override
				public int compare(SortInfomation o1, SortInfomation o2) {
					return o2.val.compareTo(o1.val);
				}
			});
			
			for(int k=0; k<K; k++){
				SortInfomation info = list.get(k);
				resMat.set(info.i, info.j, 1);
			}
		}
		
		return resMat;
	}
	
}

class SortInfomation{
	Double val;
	int i;
	int j;
	
	public SortInfomation(Double val, int i, int j) {
		super();
		this.val = val;
		this.i = i;
		this.j = j;
	}
}
