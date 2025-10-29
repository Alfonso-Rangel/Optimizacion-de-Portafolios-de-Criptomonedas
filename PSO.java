import java.util.Arrays;
import java.util.Random;

public class PSO {
    private final int maxIteraciones = 500;
    private final int maxPoblacion = 50;
    private final int dimension;
    private Particula[] poblacion;
    private Particula mejorGlobal;
    private static double[][] retornos; // [t][n]
    private double floor = 0, ceiling = 1;

    public static class Particula {
        double[] x;
        double[] mejorPosicion;
        double mejorValor;

        public Particula(int dim) {
            x = new double[dim];
            mejorPosicion = new double[dim];
        }
    }

    public PSO(double[][] retornos) {
        this.dimension = retornos[0].length;
        PSO.retornos = retornos;
    }

    // ------------------------------------------------------------
    // Funciones principales
    // ------------------------------------------------------------
    private static double calcularMedia(double[] v) {
        double s = 0;
        for (double x : v) s += x;
        return s / v.length;
    }

    private static double calcularDesviacion(double[] v, double media) {
        double s = 0;
        for (double x : v) s += Math.pow(x - media, 2);
        return Math.sqrt(s / v.length);
    }

    // Curtosis de Pearson
    public static double calcularKurtosis(double[] x) {
        double media = calcularMedia(x);
        double var = 0.0, m4 = 0.0;
        for (double v : x) {
            double diff = v - media;
            var += diff * diff;
            m4 += diff * diff * diff * diff;
        }
        var /= x.length;
        m4 /= x.length;
        if (var == 0) return 0;
        return m4 / (var * var);
    }

    // Calcula la curtosis del portafolio dado un vector de pesos
    public static double calcularKurtosisPortafolio(double[] w) {
        int n = retornos.length;
        int m = w.length;
        double[] port = new double[n];
        for (int t = 0; t < n; t++) {
            double sum = 0;
            for (int i = 0; i < m; i++) sum += w[i] * retornos[t][i];
            port[t] = sum;
        }
        return calcularKurtosis(port);
    }

    private Particula[] inicializarPoblacion() {
        Random rnd = new Random();
        Particula[] p = new Particula[maxPoblacion];
        for (int i = 0; i < maxPoblacion; i++) {
            p[i] = new Particula(dimension);
            for (int j = 0; j < dimension; j++)
                p[i].x[j] = rnd.nextDouble();
            normalizarPesos(p[i].x);
            p[i].mejorPosicion = Arrays.copyOf(p[i].x, dimension);
            p[i].mejorValor = calcularKurtosisPortafolio(p[i].x);
        }
        return p;
    }

    private void normalizarPesos(double[] x) {
        for (int i = 0; i < x.length; i++)
            x[i] = Math.max(floor, Math.min(ceiling, x[i]));
        double suma = Arrays.stream(x).sum();
        if (suma == 0) Arrays.fill(x, 1.0 / x.length);
        else for (int i = 0; i < x.length; i++) x[i] /= suma;
    }

    private Particula topologiaAnillo(int idx) {
        int izq = (idx - 1 + maxPoblacion) % maxPoblacion;
        int der = (idx + 1) % maxPoblacion;
        int[] vecinos = {izq, idx, der};
        Particula mejor = poblacion[vecinos[0]];
        for (int j : vecinos) {
            if (poblacion[j].mejorValor < mejor.mejorValor)
                mejor = poblacion[j];
        }
        return mejor;
    }

    private void actualizacionBarebones() {
        Random rnd = new Random();
        for (int i = 0; i < maxPoblacion; i++) {
            Particula mejorVecino = topologiaAnillo(i);
            for (int j = 0; j < dimension; j++) {
                double media = (poblacion[i].mejorPosicion[j] + mejorVecino.mejorPosicion[j]) / 2.0;
                double desviacion = Math.abs(poblacion[i].mejorPosicion[j] - mejorVecino.mejorPosicion[j]);
                poblacion[i].x[j] = rnd.nextGaussian() * desviacion + media;
            }
            normalizarPesos(poblacion[i].x);
        }
    }

    public double[] getMejorPosicion() {
        return Arrays.copyOf(mejorGlobal.x, mejorGlobal.x.length);
    }

    public void minimizarKurtosis() {
        poblacion = inicializarPoblacion();
        mejorGlobal = new Particula(dimension);
        mejorGlobal.x = Arrays.copyOf(poblacion[0].x, dimension);
        mejorGlobal.mejorValor = poblacion[0].mejorValor;

        for (int k = 0; k < maxIteraciones; k++) {
            for (int i = 0; i < maxPoblacion; i++) {
                double valor = calcularKurtosisPortafolio(poblacion[i].x);
                if (valor < poblacion[i].mejorValor) {
                    poblacion[i].mejorValor = valor;
                    poblacion[i].mejorPosicion = Arrays.copyOf(poblacion[i].x, dimension);
                }
                if (valor < mejorGlobal.mejorValor) {
                    mejorGlobal.x = Arrays.copyOf(poblacion[i].x, dimension);
                    mejorGlobal.mejorValor = valor;
                }
            }
            actualizacionBarebones();
        }
        System.out.println("PSO finalizado. Mejor curtosis = " + mejorGlobal.mejorValor);
    }
}