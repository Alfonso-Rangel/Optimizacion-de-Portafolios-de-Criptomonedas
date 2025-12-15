import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

public class PSO {

    private final int maxIter = 150;
    private final int nPart = 100;
    private final int dim;

    private final int tamArchivo = 100;
    private final double tasaMut = 0.6;

    private final double[][] retornos;
    private final Random rnd;

    private Particula[] pobl;
    private final ArrayList<Solucion> archivo = new ArrayList<>();

    public static class Particula {
        double[] x;
        double[] v;
        double[] pbestPos;
        double[] pbestObjMO;

        public Particula(int d) {
            x = new double[d];
            v = new double[d];
            pbestPos = new double[d];
            pbestObjMO = new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
        }
    }

    public static class Solucion {
        double[] pos;
        double[] obj;
        double crowd;

        public Solucion(double[] p, double[] o) {
            pos = p;
            obj = o;
            crowd = 0.0;
        }
    }

    // Constructor por defecto
    public PSO(double[][] ret) {
        this.retornos = ret;
        this.dim = ret[0].length;
        this.rnd = new Random(System.currentTimeMillis());
    }

    // ---------------- Multi Objectivo (MOPSO) ----------------
    public void optimizar(double[] rf) {
        initPobl();
        archivo.clear();

        for (Particula p : pobl) {
            double[] obj = evalMO(p.x, rf);
            p.pbestObjMO = Arrays.copyOf(obj, obj.length);
            p.pbestPos = Arrays.copyOf(p.x, dim);
            insertarArchivo(p.x, obj);
        }

        for (int it = 0; it < maxIter; it++) {
            calcCrowd();

            for (Particula p : pobl) {
                Solucion leader = selectLeader();
                double[] guia = (leader != null) ? leader.pos : p.pbestPos;

                mover(p, guia, 0.4, 1.8, 1.8);
                mutar(p, it);
                normalizar(p.x);

                double[] objN = evalMO(p.x, rf);
                actualizarPBestMO(p, objN);
                insertarArchivo(p.x, objN);
            }
        }
    }

    // ---------------- Helpers del PSO ----------------
    private void initPobl() {
        pobl = new Particula[nPart];
        for (int i = 0; i < nPart; i++) {
            pobl[i] = new Particula(dim);
            for (int j = 0; j < dim; j++) {
                pobl[i].x[j] = rnd.nextDouble();
                pobl[i].v[j] = 0.0;
            }
            normalizar(pobl[i].x);
        }
    }

    private void mover(Particula p, double[] lider, double w, double c1, double c2) {
        for (int j = 0; j < dim; j++) {
            double r1 = rnd.nextDouble();
            double r2 = rnd.nextDouble();
            p.v[j] = w * p.v[j]
                    + c1 * r1 * (p.pbestPos[j] - p.x[j])
                    + c2 * r2 * (lider[j] - p.x[j]);
            p.x[j] += p.v[j];
        }
    }

    private void normalizar(double[] x) {
        for (int i = 0; i < x.length; i++) {
            if (x[i] < 0.0) x[i] = 0.0;
            if (x[i] > 1.0) x[i] = 1.0;
        }
        double sum = Arrays.stream(x).sum();
        if (sum == 0.0) {
            Arrays.fill(x, 1.0 / x.length);
        } else {
            for (int i = 0; i < x.length; i++) x[i] /= sum;
        }
    }

    private void mutar(Particula p, int t) {
        double prob = Math.pow(1.0 - ((double) t / maxIter), 5.0 / tasaMut);
        if (rnd.nextDouble() < prob) {
            int d = 1 + rnd.nextInt(Math.max(1, dim / 3));
            for (int k = 0; k < d; k++) {
                int idx = rnd.nextInt(dim);
                double range = prob;
                double lb = Math.max(0.0, p.x[idx] - range);
                double ub = Math.min(1.0, p.x[idx] + range);
                p.x[idx] = lb + (ub - lb) * rnd.nextDouble();
            }
            normalizar(p.x);
        }
    }

    private double[] evalMO(double[] w, double[] rf) {
        double k = kurtosis(w);
        double s = sharpe(w, rf);
        if (Double.isNaN(k) || Double.isInfinite(k)) k = 1e6;
        if (Double.isNaN(s) || Double.isInfinite(s)) s = -1e6;
        return new double[]{k, -s};
    }

    private void actualizarPBestMO(Particula p, double[] obj) {
        int d = domina(obj, p.pbestObjMO);
        if (d == -1) {
            p.pbestObjMO = Arrays.copyOf(obj, obj.length);
            p.pbestPos = Arrays.copyOf(p.x, dim);
        } else if (d == 0 && rnd.nextBoolean()) {
            p.pbestObjMO = Arrays.copyOf(obj, obj.length);
            p.pbestPos = Arrays.copyOf(p.x, dim);
        }
    }

    private void insertarArchivo(double[] pos, double[] obj) {
        if (Double.isNaN(obj[0]) || Double.isNaN(obj[1])) return;

        boolean dominate = false;
        ArrayList<Solucion> eliminar = new ArrayList<>();

        for (Solucion s : archivo) {
            int comp = domina(s.obj, obj);
            if (comp == -1) {
                dominate = true;
                break;
            } else if (comp == 1) {
                eliminar.add(s);
            } else if (Arrays.equals(s.obj, obj)) {
                dominate = true;
                break;
            }
        }

        if (dominate) return;
        archivo.removeAll(eliminar);
        archivo.add(new Solucion(pos.clone(), obj.clone()));

        if (archivo.size() > tamArchivo) {
            controlarArchivo();
        }
    }

    private void controlarArchivo() {
        calcCrowd();
        Collections.sort(archivo, Comparator.comparingDouble(s -> s.crowd));
        archivo.remove(0);
    }

    private void calcCrowd() {
        int n = archivo.size();
        if (n == 0) return;
        for (Solucion s : archivo) s.crowd = 0.0;

        int m = 2;
        for (int j = 0; j < m; j++) {
            final int idx = j;
            archivo.sort(Comparator.comparingDouble(s -> s.obj[idx]));
            archivo.get(0).crowd = Double.POSITIVE_INFINITY;
            archivo.get(n - 1).crowd = Double.POSITIVE_INFINITY;

            double min = archivo.get(0).obj[idx];
            double max = archivo.get(n - 1).obj[idx];
            double range = max - min;
            if (range < 1e-12) range = 1.0;

            for (int i = 1; i < n - 1; i++) {
                double d = (archivo.get(i + 1).obj[idx] - archivo.get(i - 1).obj[idx]) / range;
                if (archivo.get(i).crowd != Double.POSITIVE_INFINITY) {
                    archivo.get(i).crowd += d;
                }
            }
        }
    }

    private Solucion selectLeader() {
        if (archivo.isEmpty()) return null;
        int i1 = rnd.nextInt(archivo.size());
        int i2 = rnd.nextInt(archivo.size());
        return archivo.get(i1).crowd > archivo.get(i2).crowd ? archivo.get(i1) : archivo.get(i2);
    }

    public static int domina(double[] A, double[] B) {
        boolean better = false;
        boolean worse = false;
        for (int i = 0; i < A.length; i++) {
            if (A[i] < B[i]) better = true;
            if (A[i] > B[i]) worse = true;
        }
        if (better && !worse) return -1;
        if (worse && !better) return 1;
        return 0;
    }

    public ArrayList<double[]> getFrentePos() {
        ArrayList<double[]> l = new ArrayList<>();
        for (Solucion s : archivo) l.add(s.pos.clone());
        return l;
    }

    public ArrayList<double[]> getFrenteObj() {
        ArrayList<double[]> l = new ArrayList<>();
        for (Solucion s : archivo) l.add(s.obj.clone());
        return l;
    }

    private double kurtosis(double[] w) {
        int T = retornos.length;
        double[] port = new double[T];

        for (int t = 0; t < T; t++) {
            double v = 0.0;
            for (int i = 0; i < dim; i++) v += w[i] * retornos[t][i];
            port[t] = v;
        }

        double mean = Arrays.stream(port).average().orElse(0.0);
        double m2 = 0, m4 = 0;
        for (double v : port) {
            double d = v - mean;
            double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= T;
        m4 /= T;

        if (m2 < 1e-12) return 0.0;
        return (m4 / (m2 * m2)) - 3.0;
    }

    private double sharpe(double[] w, double[] rf) {
        int T = retornos.length;
        double s1 = 0, s2 = 0;

        for (int t = 0; t < T; t++) {
            double v = 0.0;
            for (int i = 0; i < dim; i++) v += w[i] * retornos[t][i];
            double exc = v - rf[t];
            s1 += exc;
            s2 += exc * exc;
        }

        double mean = s1 / T;
        double var = (s2 / T) - mean * mean;
        if (var < 1e-12) return 0.0;
        return mean / Math.sqrt(var);
    }
}