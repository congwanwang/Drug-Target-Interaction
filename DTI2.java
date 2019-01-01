package com.qqx.cmf;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import com.qqx.common.DataProcessing;
import com.qqx.common.DataSource;
import com.qqx.common.FileUtils;

import Jama.Matrix;

public class DTI2 {

	private int rowNum = 0;
	private int colNum = 0;
	private int fDim = 0;
	private int K = 0;

	private Matrix weightMat = null;

	private List<String> truePositive = new ArrayList<>(); // 记录 1->0 的位置信息 

	public DTI2(int rowNum, int colNum, int fDim, int K) {
		super();
		this.rowNum = rowNum;
		this.colNum = colNum;
		this.fDim = fDim;
		this.K = K;

		this.weightMat = new Matrix(rowNum, colNum);
	}

	public static void main(String[] args) throws IOException {

		DTI2 dti = new DTI2(664, 445, 50, 5);
		double lambda = 0.02;
		//读取真实矩阵A
		File file = new File("C:/Data/MatrixA.txt");
		Matrix A = dti.preProcess(file);
//
		Matrix R = dti.nmf(A);
		Matrix simMat = dti.calcSimilarityMatrix(A);
		Matrix diagMat = dti.calcDiagonalMatrix(simMat);
////		simMat.print(1, 5);
//		Matrix R = dti.nmfOD(A, simMat, diagMat, lambda);
		dti.calcAUC(A, R);
	}
	
	public Matrix nmfOD(Matrix A, Matrix simMat, Matrix diagMat, double lambda) {
		// 随机定义元素为0的矩阵P和Q
		File QMatFile = new File(FileUtils.getFileAbsolutePath("data/QMat.txt"));
		File PMatFile = new File(FileUtils.getFileAbsolutePath("data/PMat.txt"));
		Matrix P = null;
		Matrix Q = null;
		if(QMatFile.exists() && PMatFile.exists()){
//			System.out.println("从文件中初始化P、Q矩阵.....");
			P = DataSource.getMatrixFromFile(PMatFile);
			Q = DataSource.getMatrixFromFile(QMatFile);
		}else{
//			System.out.println("随机初始化P、Q矩阵.....");
			P = Matrix.random(rowNum, fDim).times(0.1);
			Q = Matrix.random(fDim, colNum).times(0.1);
			
			DataSource.writeMatrixToFile(P, PMatFile);
			DataSource.writeMatrixToFile(Q, QMatFile);
		}
		
		
		// A.print(1,0);
		Matrix R = null;
		for (int c = 0; c < 150; c++) {
			Matrix Pnominator = A.times(Q.transpose());
			Matrix Pdenominator = P.times(Q).times(Q.transpose());
			modify(Pdenominator);
			Matrix U = Pnominator.arrayRightDivide(Pdenominator);
			P = P.arrayTimes(U);
			
//			Matrix Qnominator = A.transpose().times(P).plus(simMat.times(Q.transpose()).times(lambda));
//			Matrix Qdenominator = Q.transpose().times(P.transpose()).times(P)
//					.plus(diagMat.times(Q.transpose()).times(lambda));
			Matrix Qnominator = A.transpose().times(P);
			Matrix Qdenominator = Q.transpose().times(P.transpose()).times(P);
			
			modify(Qdenominator);
			Matrix K = Qnominator.arrayRightDivide(Qdenominator);
//			System.out.println(Q.getRowDimension()+" "+Q.getColumnDimension());
//			System.out.println(K.getRowDimension()+" "+K.getColumnDimension());
			Q = Q.arrayTimes(K.transpose());
			
			R = P.times(Q);
			// 计算RMSE
			double sum = 0.0;
			int n = (R.getColumnDimension()) * (R.getRowDimension());
			int zero = 0;
			for (int i = 0; i < R.getRowDimension(); i++) {
				for (int j = 0; j < R.getColumnDimension(); j++) {
					double Rij = R.get(i, j);
					double Aij = A.get(i, j);
					double tmp = Math.pow((Aij - Rij), 2);
					if (Double.isNaN(tmp)) {
						sum += 0;
						zero++;

					} else {
						sum += tmp;
					}
					// System.out.println((Aij-Rij));
					// System.out.println(sum);
				}
			}
			double RMSE = Math.sqrt(sum / n);
//			 System.out.println(RMSE);
		}
		return R;
		
	}

	public Matrix nmf(Matrix A) {
		// 随机定义元素为0的矩阵P和Q
		File QMatFile = new File(FileUtils.getFileAbsolutePath("data/QMat.txt"));
		File PMatFile = new File(FileUtils.getFileAbsolutePath("data/PMat.txt"));
		Matrix P = null;
		Matrix Q = null;
		if(QMatFile.exists() && PMatFile.exists()){
//			System.out.println("从文件中初始化P、Q矩阵.....");
			P = DataSource.getMatrixFromFile(PMatFile);
			Q = DataSource.getMatrixFromFile(QMatFile);
		}else{
//			System.out.println("随机初始化P、Q矩阵.....");
			P = Matrix.random(rowNum, fDim).times(0.1);
			Q = Matrix.random(fDim, colNum).times(0.1);
			
			DataSource.writeMatrixToFile(P, PMatFile);
			DataSource.writeMatrixToFile(Q, QMatFile);
		}

		// A.print(1,0);
		Matrix R = null;
		double lambda = 0.2;
		for (int c = 0; c < 100; c++) {

			/////// Pij和Qij更新公式
			Matrix Pnominator = A.times(Q.transpose());
			Matrix Pdenominator = P.times(lambda).plus(P.times(Q.times(Q.transpose())));
			modify(Pdenominator);
			Matrix U = Pnominator.arrayRightDivide(Pdenominator);

			Matrix Qnominator = P.transpose().times(A);
			Matrix Qdenominator = (Q.times(lambda)).plus(P.transpose().times(P.times(Q)));
			modify(Qdenominator);
			Matrix K = Qnominator.arrayRightDivide(Qdenominator);

			for (int i = 0; i < U.getRowDimension(); i++) {
				for (int j = 0; j < U.getColumnDimension(); j++) {
					double Uij = U.get(i, j);
					double Pij = P.get(i, j);
					Pij = Pij * Uij;
					P.set(i, j, Pij);
				}
			}
			for (int i = 0; i < K.getRowDimension(); i++) {
				for (int j = 0; j < K.getColumnDimension(); j++) {
					double Kij = K.get(i, j);
					double Qij = Q.get(i, j);
					Qij = Qij * Kij;
					Q.set(i, j, Qij);
				}

			}

			R = P.times(Q);
			// 计算RMSE
			double sum = 0.0;
			int n = (R.getColumnDimension()) * (R.getRowDimension());
			int zero = 0;
			for (int i = 0; i < R.getRowDimension(); i++) {
				for (int j = 0; j < R.getColumnDimension(); j++) {
					double Rij = R.get(i, j);
					double Aij = A.get(i, j);
					double tmp = Math.pow((Aij - Rij), 2);
					if (Double.isNaN(tmp)) {
						sum += 0;
						zero++;

					} else {
						sum += tmp;
					}
					// System.out.println((Aij-Rij));
					// System.out.println(sum);
				}
			}
			double RMSE = Math.sqrt(sum / n);
			// System.out.println(RMSE);

		}

		// Q.print(1, 10);
		// 计算相似度矩阵
		calcSimilarity(A, Q, K);
		return R;

	}

	public static void modify(Matrix mat) {
		for (int i = 0; i < mat.getRowDimension(); i++) {
			for (int j = 0; j < mat.getColumnDimension(); j++) {
				if (mat.get(i, j) < 1e-32) {
					mat.set(i, j, 1e-32);
				}
			}
		}
	}

	public void calcAUC(Matrix A, Matrix R) {
		// AUC计算
		///// 给预测值矩阵中90%的数按从大到小的顺序排序
		List<SortInfomation> list = new ArrayList<>();

		// 将真实矩阵中90%值为1的元素排除，其他数放入数列
		for (int i = 0; i < R.getRowDimension(); i++) {
			for (int j = 0; j < R.getColumnDimension(); j++) {
				if (A.get(i, j) == 1) {
					continue;
				}

				 list.add(new SortInfomation(R.get(i, j), i, j));
//				list.add(new SortInfomation(R.get(i, j) * (weightMat.get(i, j) / (K)), i, j));
			}
		}

		Collections.sort(list, new Comparator<SortInfomation>() {
			@Override
			public int compare(SortInfomation o1, SortInfomation o2) {
				return o2.val.compareTo(o1.val);
			}
		});
		
		System.out.println("靠前的预测结果：");
		for(int i=0; i<40; i++){
			System.out.println(list.get(i).val);
		}
		
		// 以阈值为基准将tempc分成两个submap并将其中的值与位置对应
		int TP = 0;
		int FN = 0;
		int FP = 0;
		int TN = 0;
		for (int k = 0; k < list.size(); k++) {
			SortInfomation info = list.get(k);
			int ai = info.i;
			int aj = info.j;

			// 计算TP，FN，FP，TN的个数
			if (truePositive.contains(ai + "#" + aj)) {
				TP++;
			} else {
				FP++;
			}
			FN = truePositive.size() - TP;
			TN = list.size() - TP - FP - FN;

			double TPR = (double) TP / (TP + FN);
			double FPR = (double) FP / (FP + TN);
			double Recall = (double) TP / (TP + FN);
			double Precision = (double) TP / (FP + TP);

			if (k % 10 == 0) {
				// if (FPR < 0) {
				// System.out.println("出现错误：" + FP + " , " + TN);
				// }
//				System.out.println(FPR + ":" + TPR);
//				System.out.println(Recall+":"+Precision);
			}
		}
	}

	/**
	 * 预处理
	 * 
	 * @param file
	 *            数据集文件
	 * @return
	 */
	public Matrix preProcess(File file) {
		Matrix A = DataProcessing.txt2String(file);
		// mat.print(1, 0);
		File testSampleFile = new File(FileUtils.getFileAbsolutePath("data/testData.txt"));
		truePositive = FileUtils.readFileByLine(testSampleFile);
		for(String str : truePositive){
			String[] ijArr = str.split("#");
			int i = Integer.parseInt(ijArr[0]);
			int j = Integer.parseInt(ijArr[1]);
			A.set(i, j, 0);
		}
		return A;
	}

	public static double modifyNan(double num1, double num2, double multiplier) {
		double res = num1 * multiplier;
		res *= (num2 * multiplier);
		if (Double.isNaN(res)) {
			return 1e-32;
		}
		return res;
	}

	public Matrix calcSimilarityMatrix(Matrix A) {
		int col = A.getColumnDimension();
		int row = A.getRowDimension();
		Matrix simMat = new Matrix(col, col);
		for (int i = 0; i < col; i++) {
			for (int j = i + 1; j < col; j++) {
				double top = 0.0;
				double bottomL = 0.0;
				double bottomR = 0.0;

				// 计算平均值
				double iAvg = 0.0;
				double jAvg = 0.0;
				for (int d = 0; d < row; d++) {
					iAvg += A.get(d, i);
					jAvg += A.get(d, j);
				}
				iAvg /= row;
				jAvg /= row;

				for (int d = 0; d < row; d++) {
					top += (A.get(d, i) - iAvg) * (A.get(d, j) - jAvg);
					bottomL += Math.pow(A.get(d, i) - iAvg, 2);
					bottomR += Math.pow(A.get(d, j) - jAvg, 2);
//					 top += modifyNan(A.get(d, i), A.get(d, j), 100);
//					 bottomL += modifyNan(A.get(d, i), A.get(d, i),100);
//					 bottomR += modifyNan(A.get(d, j), A.get(d, j),100);

				}

				// if(i==0 && j==4){
				// System.out.println(top+" "+bottomL+" "+bottomR);
				// }
				double sim = 0;
				if (top != 0 && bottomL != 0 && bottomR != 0) {
					sim = top / Math.pow(bottomL * bottomR, 0.5);
				}

				// if(sim == 1 && time < 6){
				// for(int d=0; d<f; d++){
				// System.out.println(QMat.get(d, i)+" : "+QMat.get(d, j));
				// }
				// System.out.println(top+" "+bottomL+" "+bottomR);
				// System.out.println();
				// time ++;
				// }
				double sigmodVal = 1.0/(1.0+Math.pow(Math.E, -sim));
				if(sigmodVal > 0.5){
					simMat.set(i, j, 1);
					simMat.set(j, i, 1);
				}else{
					simMat.set(i, j, 0);
					simMat.set(j, i, 0);
				}
			}
			// return simMat;
		}
		return simMat;
	}
	
	public Matrix calcDiagonalMatrix (Matrix simMat){
		int n = simMat.getColumnDimension();
		Matrix diagMat = new Matrix(n, n);
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int j = 0; j < n; j++) {
				sum += simMat.get(i, j);
			}
			diagMat.set(i, i, sum);
		}
		return diagMat;
	}
	/**
	 * 
	 * @param QMat
	 * @param topK
	 *            取前topK个最相似
	 */
	public void calcSimilarity(Matrix A, Matrix QMat, int topK) {
		int n = QMat.getColumnDimension();
		int m = A.getRowDimension();
		int f = QMat.getRowDimension();
		Matrix simMat = new Matrix(n, n);
		int time = 0;
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				double top = 0.0;
				double bottomL = 0.0;
				double bottomR = 0.0;

				// 计算平均值
				double iAvg = 0.0;
				double jAvg = 0.0;
				for (int d = 0; d < f; d++) {
					iAvg += QMat.get(d, i);
					jAvg += QMat.get(d, j);
				}
				iAvg /= f;
				jAvg /= f;

				for (int d = 0; d < f; d++) {
					top += (QMat.get(d, i) - iAvg) * (QMat.get(d, j) - jAvg);
					bottomL += Math.pow(QMat.get(d, i) - iAvg, 2);
					bottomR += Math.pow(QMat.get(d, j) - jAvg, 2);
					// top += modifyNan(QMat.get(d, i), QMat.get(d, j), 100);
					// bottomL += modifyNan(QMat.get(d, i), QMat.get(d, i),
					// 100);
					// bottomR += modifyNan(QMat.get(d, j), QMat.get(d, j),
					// 100);

				}

				// if(i==0 && j==4){
				// System.out.println(top+" "+bottomL+" "+bottomR);
				// }
				double sim = 0;
				if (top != 0 && bottomL != 0 && bottomR != 0) {
					sim = top / Math.pow(bottomL * bottomR, 0.5);
				}

				// if(sim == 1 && time < 6){
				// for(int d=0; d<f; d++){
				// System.out.println(QMat.get(d, i)+" : "+QMat.get(d, j));
				// }
				// System.out.println(top+" "+bottomL+" "+bottomR);
				// System.out.println();
				// time ++;
				// }
				simMat.set(i, j, sim);
				simMat.set(j, i, sim);
			}
		}
		// simMat.print(1,4);
		// System.out.println(simMat.get(0, 4));

		for (int i = 0; i < n; i++) {
			List<SortInfomation> list = new ArrayList<>();
			for (int j = 0; j < n; j++) {
				list.add(new SortInfomation(simMat.get(i, j), i, j));
			}

			Collections.sort(list, new Comparator<SortInfomation>() {
				@Override
				public int compare(SortInfomation o1, SortInfomation o2) {
					return o2.val.compareTo(o1.val);
				}
			});

			for (int k = 0; k < m; k++) {
				double weight = 0.0;
				for (int j = 0; j < topK; j++) {
					SortInfomation info = list.get(j);
					weight += info.val * A.get(info.i, i);
				}

				weightMat.set(k, i, weight);
			}
		}

	}

}

class SortInfo {
	Double val;
	int i;
	int j;

	public SortInfo(Double val, int i, int j) {
		super();
		this.val = val;
		this.i = i;
		this.j = j;
	}

}
