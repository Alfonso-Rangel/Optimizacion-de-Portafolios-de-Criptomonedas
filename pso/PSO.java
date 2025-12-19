import java.util.ArrayList;
import java.util.Comparator;
import java.util.Random;

public class PSO {

    private final int maxIter = 150;
    private final int nPart = 100;
    private final int dim;
    private final int tamArchivo = 100;
    private final double tasaMut = 0.6;

    private final double[][] retornos;
    private final int T;
    private final Random rnd;

    private Particula[] pobl;
    private final ArrayList<Solucion> archivo = new ArrayList<>(tamArchivo + 10);

    // ---------------- Estructuras ----------------

    static class Particula {
        double[] x, v, pbestPos;
        double[] pbestObj;

        Particula(int d) {
            x = new double[d];
            v = new double[d];
            pbestPos = new double[d];
            pbestObj = new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
        }
    }

    static class Solucion {
        double[] pos;
        double[] obj;
        double crowd;

        Solucion(int d) {
            pos = new double[d];
            obj = new double[2];
        }
    }

    // ---------------- Constructor ----------------

    public PSO(double[][] ret) {
        this.retornos = ret;
        this.T = ret.length;
        this.dim = ret[0].length;
        this.rnd = new Random(123456);
    }

    // ---------------- Optimización ----------------

    public void optimizar(double[] rf) {
        initPobl();
        archivo.clear();

        for (Particula p : pobl) {
            evalMO(p.x, p.pbestObj, rf);
            System.arraycopy(p.x, 0, p.pbestPos, 0, dim);
            insertarArchivo(p.x, p.pbestObj);
        }

        for (int it = 0; it < maxIter; it++) {
            calcCrowd();

            for (Particula p : pobl) {
                Solucion leader = selectLeader();
                double[] guia = leader != null ? leader.pos : p.pbestPos;

                mover(p, guia, 0.4, 1.8, 1.8);
                mutar(p, it);
                normalizar(p.x);

                double[] obj = new double[2];
                evalMO(p.x, obj, rf);
                actualizarPBest(p, obj);
                insertarArchivo(p.x, obj);
            }
        }
    }

    // ---------------- PSO ----------------

    private void initPobl() {
        pobl = new Particula[nPart];
        for (int i = 0; i < nPart; i++) {
            Particula p = new Particula(dim);
            for (int j = 0; j < dim; j++) {
                p.x[j] = rnd.nextDouble();
                p.v[j] = 0.0;
            }
            normalizar(p.x);
            pobl[i] = p;
        }
    }

    private void mover(Particula p, double[] g, double w, double c1, double c2) {
        for (int j = 0; j < dim; j++) {
            double r1 = rnd.nextDouble();
            double r2 = rnd.nextDouble();
            p.v[j] = w * p.v[j]
                    + c1 * r1 * (p.pbestPos[j] - p.x[j])
                    + c2 * r2 * (g[j] - p.x[j]);
            p.x[j] += p.v[j];
        }
    }

    private void normalizar(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            if (x[i] < 0.0) x[i] = 0.0;
            else if (x[i] > 1.0) x[i] = 1.0;
            sum += x[i];
        }
        if (sum == 0.0) {
            double v = 1.0 / dim;
            for (int i = 0; i < dim; i++) x[i] = v;
        } else {
            double inv = 1.0 / sum;
            for (int i = 0; i < dim; i++) x[i] *= inv;
        }
    }

    // ---------------- Mutación ----------------

    private void mutar(Particula p, int t) {
        double prob = Math.pow(1.0 - (double) t / maxIter, 5.0 / tasaMut);
        if (rnd.nextDouble() < prob) {
            int d = 1 + rnd.nextInt(Math.max(1, dim / 3));
            for (int k = 0; k < d; k++) {
                int idx = rnd.nextInt(dim);
                double range = prob;
                double lb = p.x[idx] - range;
                double ub = p.x[idx] + range;
                if (lb < 0.0) lb = 0.0;
                if (ub > 1.0) ub = 1.0;
                p.x[idx] = lb + (ub - lb) * rnd.nextDouble();
            }
            normalizar(p.x);
        }
    }

    // ---------------- Evaluación ----------------

    private void evalMO(double[] w, double[] out, double[] rf) {
        double mean = 0.0;
        double m2 = 0.0;
        double m4 = 0.0;
        double sharpeNum = 0.0;
        double sharpeDen = 0.0;

        for (int t = 0; t < T; t++) {
            double v = 0.0;
            for (int i = 0; i < dim; i++) {
                v += w[i] * retornos[t][i];
            }
            mean += v;
            double exc = v - rf[t];
            sharpeNum += exc;
            sharpeDen += exc * exc;
        }

        mean /= T;

        for (int t = 0; t < T; t++) {
            double v = 0.0;
            for (int i = 0; i < dim; i++) {
                v += w[i] * retornos[t][i];
            }
            double d = v - mean;
            double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }

        m2 /= T;
        m4 /= T;

        double kurt = (m2 < 1e-12) ? 0.0 : (m4 / (m2 * m2) - 3.0);
        double var = sharpeDen / T - Math.pow(sharpeNum / T, 2);
        double sharpe = (var < 1e-12) ? 0.0 : (sharpeNum / T) / Math.sqrt(var);

        out[0] = kurt;
        out[1] = -sharpe;
    }

    // ---------------- Archivo ----------------

    private void insertarArchivo(double[] pos, double[] obj) {
        for (int i = archivo.size() - 1; i >= 0; i--) {
            int d = domina(archivo.get(i).obj, obj);
            if (d == -1) return;
            if (d == 1) archivo.remove(i);
        }

        Solucion s = new Solucion(dim);
        System.arraycopy(pos, 0, s.pos, 0, dim);
        s.obj[0] = obj[0];
        s.obj[1] = obj[1];
        archivo.add(s);

        if (archivo.size() > tamArchivo) {
            calcCrowd();
            archivo.sort(Comparator.comparingDouble(a -> a.crowd));
            archivo.remove(0);
        }
    }

    private void calcCrowd() {
        int n = archivo.size();
        if (n < 3) return;

        for (Solucion s : archivo) s.crowd = 0.0;

        for (int m = 0; m < 2; m++) {
            final int idx = m;
            archivo.sort(Comparator.comparingDouble(s -> s.obj[idx]));

            archivo.get(0).crowd = Double.POSITIVE_INFINITY;
            archivo.get(n - 1).crowd = Double.POSITIVE_INFINITY;

            double min = archivo.get(0).obj[idx];
            double max = archivo.get(n - 1).obj[idx];
            double range = max - min;
            if (range < 1e-12) range = 1.0;

            for (int i = 1; i < n - 1; i++) {
                archivo.get(i).crowd +=
                        (archivo.get(i + 1).obj[idx] - archivo.get(i - 1).obj[idx]) / range;
            }
        }
    }

    private Solucion selectLeader() {
        int a = rnd.nextInt(archivo.size());
        int b = rnd.nextInt(archivo.size());
        return archivo.get(a).crowd > archivo.get(b).crowd ? archivo.get(a) : archivo.get(b);
    }

    // ---------------- Dominancia ----------------

    static int domina(double[] A, double[] B) {
        boolean better = false, worse = false;
        for (int i = 0; i < A.length; i++) {
            if (A[i] < B[i]) better = true;
            else if (A[i] > B[i]) worse = true;
        }
        if (better && !worse) return -1;
        if (worse && !better) return 1;
        return 0;
    }

    private void actualizarPBest(Particula p, double[] obj) {
        int d = domina(obj, p.pbestObj);
        if (d == -1 || (d == 0 && rnd.nextBoolean())) {
            System.arraycopy(obj, 0, p.pbestObj, 0, 2);
            System.arraycopy(p.x, 0, p.pbestPos, 0, dim);
        }
    }

    public ArrayList<double[]> getFrentePos() {
    ArrayList<double[]> l = new ArrayList<>(archivo.size());
    for (Solucion s : archivo) {
        double[] p = new double[dim];
        System.arraycopy(s.pos, 0, p, 0, dim);
        l.add(p);
    }
    return l;
}

    public ArrayList<double[]> getFrenteObj() {
        ArrayList<double[]> l = new ArrayList<>(archivo.size());
        for (Solucion s : archivo) {
            double[] o = new double[2];
            o[0] = s.obj[0];
            o[1] = s.obj[1];
            l.add(o);
        }
        return l;
    }

}
