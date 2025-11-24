import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

public class MOPSO {

    // --- Parámetros ---
    private final int maxIteraciones = 250; 
    private final int maxPoblacion = 100;
    private final int tamanoArchivo = 100;
    
    // Tasa de mutación
    private final double tasaMutacion = 0.6; 

    private final int dimension;
    private final Random rnd = new Random();
    private static double[][] retornos;
    
    private final ArrayList<Solucion> archivo = new ArrayList<>();

    private final double floor = 0.0;
    private final double ceiling = 1.0;

    // Clase interna simplificada
    public static class Solucion {
        double[] pos;
        double[] obj;        // [0]: Curtosis, [1]: -Sharpe
        double crowdingDist; // Distancia de aglomeración

        public Solucion(double[] pos, double[] obj) {
            this.pos = pos;
            this.obj = obj;
            this.crowdingDist = 0.0;
        }
    }

    public static class Particula {
        double[] x;
        double[] v;
        double[] pbestPos;
        double[] pbestObj;

        public Particula(int dim) {
            x = new double[dim];
            v = new double[dim];
            pbestPos = new double[dim];
            // Inicializamos pbest con valores muy malos (infinito)
            pbestObj = new double[]{Double.MAX_VALUE, Double.MAX_VALUE}; 
        }
    }

    public MOPSO(double[][] retornosLocal) {
        retornos = retornosLocal;
        this.dimension = retornosLocal[0].length;
    }

    // ============================================================
    //  Ciclo de Optimización
    // ============================================================

    public void optimizar(double[] rf) {
        Particula[] poblacion = inicializar();

        // 1. Evaluación inicial
        for (Particula p : poblacion) {
            double[] obj = evaluar(p.x, rf);
            actualizarPbest(p, obj);
            agregarAlArchivo(p.x, obj);
        }

        // Ciclo principal
        for (int t = 0; t < maxIteraciones; t++) {
            
            // Calcular distancias al inicio de la iteración para tener líderes actualizados
            if (!archivo.isEmpty()) {
                calcularCrowdingDistances();
            }

            for (int i = 0; i < maxPoblacion; i++) {
                Particula p = poblacion[i];

                // a. Selección de Líder (Torneo basado en Crowding Distance)
                Solucion lider = seleccionarLider();
                
                // Fallback si el archivo está vacío (raro)
                double[] guia = (lider != null) ? lider.pos : p.pbestPos;

                // b. Actualización
                actualizarParticula(p, guia);
                
                // c. Mutación (Turbulencia)
                aplicarMutacion(p, t);

                // d. Aplica las restricciones
                normalizar(p.x);

                // e. Evaluación
                double[] nuevosObj = evaluar(p.x, rf);

                // f. Actualizar PBest y Archivo
                actualizarPbest(p, nuevosObj);
                agregarAlArchivo(p.x, nuevosObj);
            }
        }
    }

    private void aplicarLimites(Particula p) {
        for (int j = 0; j < dimension; j++) {
            if (p.x[j] < floor) {
                p.x[j] = floor;
                p.v[j] *= -1; // Rebote
            }
            if (p.x[j] > ceiling) {
                p.x[j] = ceiling;
                p.v[j] *= -1; // Rebote
            }
        }
        normalizar(p.x);
    }
    
    private void normalizar(double[] x) {
        for (int i = 0; i < x.length; i++) x[i] = Math.max(floor, Math.min(ceiling, x[i]));
        double suma = Arrays.stream(x).sum();
        if (suma == 0) Arrays.fill(x, 1.0 / x.length);
        else for (int i = 0; i < x.length; i++) x[i] /= suma;
    }

    // ============================================================
    //  Manejo del Archivo con Crowding Distance
    // ============================================================

    private void agregarAlArchivo(double[] pos, double[] obj) {
        if (Double.isNaN(obj[0]) || Double.isNaN(obj[1])) return; 

        boolean esDominado = false;
        ArrayList<Solucion> aEliminar = new ArrayList<>();

        for (Solucion s : archivo) {
            int flag = domina(s.obj, obj);
            if (flag == -1) { // s domina a la nueva
                esDominado = true;
                break; 
            } else if (flag == 1) { // nueva domina a s
                aEliminar.add(s);
            } else if (Arrays.equals(s.obj, obj)) {
                esDominado = true; // Evitar duplicados exactos
                break;
            }
        }

        if (esDominado) return;

        archivo.removeAll(aEliminar);
        archivo.add(new Solucion(pos.clone(), obj.clone()));

        // Si superamos la capacidad, eliminamos usando Crowding Distance
        if (archivo.size() > tamanoArchivo) {
            controlarTamanoArchivo();
        }
    }

    /**
     * Calcula la métrica Crowding Distance para todas las soluciones del archivo.
     * Es inmune a la escala de los objetivos.
     */
    private void calcularCrowdingDistances() {
        int n = archivo.size();
        if (n == 0) return;

        // Reiniciar distancias
        for (Solucion s : archivo) s.crowdingDist = 0.0;

        // Para cada objetivo (0: Curtosis, 1: -Sharpe)
        int numObj = 2; 
        for (int m = 0; m < numObj; m++) {
            // 1. Ordenar archivo por el objetivo actual 'm'
            final int objIndex = m;
            Collections.sort(archivo, Comparator.comparingDouble(s -> s.obj[objIndex]));

            // 2. Asignar infinito a los extremos (para conservarlos siempre)
            archivo.get(0).crowdingDist = Double.POSITIVE_INFINITY;
            archivo.get(n - 1).crowdingDist = Double.POSITIVE_INFINITY;

            // 3. Calcular rango del objetivo
            double minVal = archivo.get(0).obj[objIndex];
            double maxVal = archivo.get(n - 1).obj[objIndex];
            double rango = maxVal - minVal;
            if (rango == 0) rango = 1.0; // Evitar división por cero

            // 4. Calcular distancia para los puntos intermedios
            for (int i = 1; i < n - 1; i++) {
                if (archivo.get(i).crowdingDist != Double.POSITIVE_INFINITY) {
                    double distance = (archivo.get(i + 1).obj[objIndex] - archivo.get(i - 1).obj[objIndex]) / rango;
                    archivo.get(i).crowdingDist += distance;
                }
            }
        }
    }

    /**
     * Elimina la solución con la MENOR distancia de aglomeración (la más "apretada").
     */
    private void controlarTamanoArchivo() {
        calcularCrowdingDistances();

        // Ordenar por distancia de menor a mayor
        // Los de menor distancia están en zonas muy pobladas -> candidatos a eliminar
        // Los de distancia infinita (extremos) quedan al final.
        Collections.sort(archivo, Comparator.comparingDouble(s -> s.crowdingDist));

        // Eliminar el primero (el más aglomerado)
        // Nota: remove(0) es eficiente en ArrayList pequeños, aunque desplaza elementos. 
        // Para tamanoArchivo=100 es despreciable.
        archivo.remove(0);
    }

    /**
     * Selección de líder por Torneo Binario.
     * Buscamos maximizar la Crowding Distance (ir a zonas menos pobladas).
     */
    private Solucion seleccionarLider() {
        if (archivo.isEmpty()) return null;
        
        int idx1 = rnd.nextInt(archivo.size());
        int idx2 = rnd.nextInt(archivo.size());

        Solucion s1 = archivo.get(idx1);
        Solucion s2 = archivo.get(idx2);

        // Retornar el que tenga MAYOR distancia (menos aglomerado)
        // Si uno es infinito (extremo), ese gana automáticamente.
        return (s1.crowdingDist > s2.crowdingDist) ? s1 : s2;
    }

    // ============================================================
    //  Dinámica de Partículas
    // ============================================================

    private void actualizarParticula(Particula p, double[] lider) {
        double w = 0.4; 
        double c1 = 1.8; 
        double c2 = 1.8; 

        for (int j = 0; j < dimension; j++) {
            double r1 = rnd.nextDouble();
            double r2 = rnd.nextDouble();
            
            p.v[j] = w * p.v[j] 
                   + c1 * r1 * (p.pbestPos[j] - p.x[j]) 
                   + c2 * r2 * (lider[j] - p.x[j]);
            
            p.x[j] += p.v[j];
        }
    }

    private void aplicarMutacion(Particula p, int t) {
        // Probabilidad decreciente
        double prob = Math.pow(1.0 - (double)t / maxIteraciones, 5.0 / tasaMutacion);
        
        if (rnd.nextDouble() < prob) {
            // Mutar una porción de las dimensiones
            int dimsAMutar = 1 + rnd.nextInt(Math.max(1, dimension / 3));
            
            for(int k=0; k<dimsAMutar; k++) {
                int dim = rnd.nextInt(dimension);
                double range = (ceiling - floor) * prob; 
                
                double lb = Math.max(floor, p.x[dim] - range);
                double ub = Math.min(ceiling, p.x[dim] + range);
                
                p.x[dim] = lb + (ub - lb) * rnd.nextDouble();
            }
            normalizar(p.x);
        }
    }

    // ============================================================
    //  Evaluación y Utilidades
    // ============================================================

    private double[] evaluar(double[] w, double[] rf) {
        double k = kurtosisPortafolio(w);
        double s = sharpePortafolio(w, rf);
        
        // Penalizaciones
        if (Double.isNaN(k) || Double.isInfinite(k)) k = 1e6; 
        if (Double.isNaN(s) || Double.isInfinite(s)) s = -1e6; 

        return new double[]{ k, -s }; 
    }

    public static int domina(double[] A, double[] B) {
        if (A == null || B == null) return 0;
        boolean mejorEnAlgo = false;
        boolean peorEnAlgo = false;

        for (int i = 0; i < A.length; i++) {
            if (A[i] < B[i]) mejorEnAlgo = true;
            if (A[i] > B[i]) peorEnAlgo = true;
        }

        if (mejorEnAlgo && !peorEnAlgo) return -1; // A domina B
        if (peorEnAlgo && !mejorEnAlgo) return 1;  // B domina A
        return 0;
    }

    private Particula[] inicializar() {
        Particula[] p = new Particula[maxPoblacion];
        for (int i = 0; i < maxPoblacion; i++) {
            p[i] = new Particula(dimension);
            for (int j = 0; j < dimension; j++) {
                p[i].x[j] = rnd.nextDouble();
                p[i].v[j] = 0.0;
            }
            normalizar(p[i].x);
        }
        return p;
    }

    private void actualizarPbest(Particula p, double[] obj) {
        int d = domina(obj, p.pbestObj);
        if (d == -1) {
            p.pbestObj = obj;
            p.pbestPos = Arrays.copyOf(p.x, dimension);
        } else if (d == 0 && rnd.nextBoolean()) {
            p.pbestObj = obj;
            p.pbestPos = Arrays.copyOf(p.x, dimension);
        }
    }

    // --- Estadísticas Vectorizadas ---
    private static double kurtosisPortafolio(double[] w) {
        int n = retornos.length;
        int m = w.length;
        double[] port = new double[n];
        double s1 = 0, s2 = 0, s4 = 0;

        for (int t = 0; t < n; t++) {
            double val = 0;
            for (int i = 0; i < m; i++) val += w[i] * retornos[t][i];
            port[t] = val;
            s1 += val;
        }
        
        double mean = s1 / n;
        for (double val : port) {
            double d = val - mean;
            double d2 = d * d;
            s2 += d2;
            s4 += d2 * d2;
        }
        
        s2 /= n;
        s4 /= n;
        
        if (s2 < 1e-9) return 0; 
        return (s4 / (s2 * s2)) - 3.0; 
    }

    private static double sharpePortafolio(double[] w, double[] rf) {
        int n = retornos.length;
        int m = w.length;
        double s1 = 0, s2 = 0;
        
        for (int t = 0; t < n; t++) {
            double val = 0;
            for (int i = 0; i < m; i++) val += w[i] * retornos[t][i];
            double excess = val - rf[t];
            s1 += excess;
            s2 += excess * excess;
        }
        
        double mean = s1 / n;
        double variance = (s2 / n) - (mean * mean);
        double std = Math.sqrt(Math.max(0, variance));
        
        if (std < 1e-9) return 0;
        return mean / std;
    }

    // Getters
    public ArrayList<double[]> getFrentePos() {
        ArrayList<double[]> list = new ArrayList<>();
        for (Solucion s : archivo) list.add(s.pos);
        return list;
    }

    public ArrayList<double[]> getFrenteObj() {
        ArrayList<double[]> list = new ArrayList<>();
        for (Solucion s : archivo) list.add(s.obj);
        return list;
    }
}