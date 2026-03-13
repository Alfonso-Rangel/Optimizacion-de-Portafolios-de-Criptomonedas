import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

public class NSGAII {

    private final int maxGen = 150;
    private final int popSize = 100; // N
    private final int dim;
    private final double pc = 0.9;  // Probabilidad de cruce
    private final double pm;        // Probabilidad de mutación (1/dim)
    private final double etaC = 5.0; // Índice para SBX
    private final double etaM = 5.0; // Índice para Mutación Polinomial

    private final double[][] retornos;
    private final int T;
    private final Random rnd = new Random();

    private ArrayList<Individuo> poblacion;

    // ---------------- Estructura del Individuo ----------------

    static class Individuo {
        double[] x;
        double[] obj;
        double crowdDist;
        int rank;
        int dominationCount;
        ArrayList<Individuo> dominatedSet = new ArrayList<>();

        Individuo(int d) {
            x = new double[d];
            obj = new double[2];
        }

        void reset() {
            dominationCount = 0;
            dominatedSet.clear();
        }
    }

    public NSGAII(double[][] ret) {
        this.retornos = ret;
        this.T = ret.length;
        this.dim = ret[0].length;
        this.pm = 1.0 / dim;
    }

    // ---------------- Algoritmo Principal ----------------

    public void optimizar(double[] rf) {
        // 1. Inicialización
        poblacion = initPobl(rf);

        for (int gen = 0; gen < maxGen; gen++) {
            // 2. Crear descendencia (Qt)
            ArrayList<Individuo> hijos = generarHijos(rf);

            // 3. Unión (Rt = Pt + Qt)
            ArrayList<Individuo> union = new ArrayList<>(poblacion);
            union.addAll(hijos);

            // 4. Clasificación No Dominada y Selección de Supervivientes
            poblacion = seleccionarSiguienteGeneracion(union);
        }
    }

    private ArrayList<Individuo> initPobl(double[] rf) {
        ArrayList<Individuo> pop = new ArrayList<>();
        for (int i = 0; i < popSize; i++) {
            Individuo ind = new Individuo(dim);
            for (int j = 0; j < dim; j++) ind.x[j] = rnd.nextDouble();
            normalizar(ind.x);
            evalMO(ind.x, ind.obj, rf);
            pop.add(ind);
        }
        return pop;
    }

    // ---------------- Operadores Genéticos (SBX y Mutación) ----------------

    private ArrayList<Individuo> generarHijos(double[] rf) {
        ArrayList<Individuo> hijos = new ArrayList<>();
        while (hijos.size() < popSize) {
            // Selección por Torneo Binario
            Individuo p1 = torneo();
            Individuo p2 = torneo();

            Individuo c1 = new Individuo(dim);
            Individuo c2 = new Individuo(dim);

            // Cruce SBX
            if (rnd.nextDouble() < pc) {
                for (int j = 0; j < dim; j++) {
                    if (rnd.nextDouble() <= 0.5) {
                        double u = rnd.nextDouble();
                        double beta = (u <= 0.5) ? Math.pow(2 * u, 1.0 / (etaC + 1)) : Math.pow(1.0 / (2 * (1 - u)), 1.0 / (etaC + 1));
                        c1.x[j] = 0.5 * ((1 + beta) * p1.x[j] + (1 - beta) * p2.x[j]);
                        c2.x[j] = 0.5 * ((1 - beta) * p1.x[j] + (1 + beta) * p2.x[j]);
                    } else {
                        c1.x[j] = p1.x[j];
                        c2.x[j] = p2.x[j];
                    }
                }
            } else {
                c1.x = p1.x.clone();
                c2.x = p2.x.clone();
            }

            // Mutación Polinomial y Evaluación
            mutar(c1); normalizar(c1.x); evalMO(c1.x, c1.obj, rf);
            mutar(c2); normalizar(c2.x); evalMO(c2.x, c2.obj, rf);

            hijos.add(c1);
            if (hijos.size() < popSize) hijos.add(c2);
        }
        return hijos;
    }

    private void mutar(Individuo ind) {
        for (int j = 0; j < dim; j++) {
            if (rnd.nextDouble() < pm) {
                double u = rnd.nextDouble();
                double delta = (u < 0.5) ? Math.pow(2 * u, 1.0 / (etaM + 1)) - 1 : 1 - Math.pow(2 * (1 - u), 1.0 / (etaM + 1));
                ind.x[j] += delta;
                if (ind.x[j] < 0) ind.x[j] = 0;
                if (ind.x[j] > 1) ind.x[j] = 1;
            }
        }
    }

    // ---------------- Lógica de Selección NSGA-II ----------------

    private ArrayList<Individuo> seleccionarSiguienteGeneracion(ArrayList<Individuo> union) {
        ArrayList<ArrayList<Individuo>> frentes = fastNonDominatedSort(union);
        ArrayList<Individuo> siguienteGen = new ArrayList<>();
        
        for (ArrayList<Individuo> frente : frentes) {
            calcularCrowdingDistance(frente);
            if (siguienteGen.size() + frente.size() <= popSize) {
                siguienteGen.addAll(frente);
            } else {
                // El último frente se ordena por Crowding Distance (descendente)
                frente.sort((a, b) -> Double.compare(b.crowdDist, a.crowdDist));
                for (int i = 0; siguienteGen.size() < popSize; i++) {
                    siguienteGen.add(frente.get(i));
                }
                break;
            }
        }
        return siguienteGen;
    }

    private ArrayList<ArrayList<Individuo>> fastNonDominatedSort(ArrayList<Individuo> pobl) {
        ArrayList<ArrayList<Individuo>> frentes = new ArrayList<>();
        ArrayList<Individuo> f1 = new ArrayList<>();

        for (Individuo p : pobl) {
            p.reset();
            for (Individuo q : pobl) {
                int d = domina(p.obj, q.obj);
                if (d == -1) p.dominatedSet.add(q);
                else if (d == 1) p.dominationCount++;
            }
            if (p.dominationCount == 0) {
                p.rank = 1;
                f1.add(p);
            }
        }
        frentes.add(f1);

        int i = 0;
        while (i < frentes.size() && !frentes.get(i).isEmpty()) {
            ArrayList<Individuo> proximoFrente = new ArrayList<>();
            for (Individuo p : frentes.get(i)) {
                for (Individuo q : p.dominatedSet) {
                    q.dominationCount--;
                    if (q.dominationCount == 0) {
                        q.rank = i + 2;
                        proximoFrente.add(q);
                    }
                }
            }
            if (!proximoFrente.isEmpty()) frentes.add(proximoFrente);
            i++;
        }
        return frentes;
    }

    private void calcularCrowdingDistance(ArrayList<Individuo> frente) {
        int n = frente.size();
        if (n == 0) return;
        for (Individuo ind : frente) ind.crowdDist = 0;

        for (int m = 0; m < 2; m++) {
            final int objIdx = m;
            frente.sort(Comparator.comparingDouble(a -> a.obj[objIdx]));
            
            frente.get(0).crowdDist = Double.POSITIVE_INFINITY;
            frente.get(n - 1).crowdDist = Double.POSITIVE_INFINITY;

            double range = frente.get(n - 1).obj[objIdx] - frente.get(0).obj[objIdx];
            if (range == 0) continue;

            for (int i = 1; i < n - 1; i++) {
                frente.get(i).crowdDist += (frente.get(i + 1).obj[objIdx] - frente.get(i - 1).obj[objIdx]) / range;
            }
        }
    }

    private Individuo torneo() {
        Individuo a = poblacion.get(rnd.nextInt(popSize));
        Individuo b = poblacion.get(rnd.nextInt(popSize));
        if (a.rank < b.rank) return a;
        if (b.rank < a.rank) return b;
        return (a.crowdDist > b.crowdDist) ? a : b;
    }

    // ---------------- Utilidades (Igual que en tu PSO) ----------------
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

    public ArrayList<double[]> getFrenteObj() {
        ArrayList<double[]> l = new ArrayList<>();
        for (Individuo ind : poblacion) {
            if (ind.rank == 1) l.add(ind.obj.clone());
        }
        return l;
    }

    public ArrayList<double[]> getFrentePos() {
        ArrayList<double[]> l = new ArrayList<>();
        for (Individuo ind : poblacion) {
            if (ind.rank == 1) l.add(ind.x.clone());
        }
        return l;
    }
}