import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

public class PSO {

    // ============================================================
    //  Configuración y Parámetros
    // ============================================================
    private final int maxIteraciones = 250;
    private final int maxPoblacion = 100;
    private final int dimension;
    
    // Parámetros MOPSO
    private final int tamanoArchivo = 100;
    private final double tasaMutacion = 0.6;

    // Datos
    private static double[][] retornos;
    private final Random rnd = new Random();
    private final double floor = 0.0;
    private final double ceiling = 1.0;

    // Estado Single-Objective (SO)
    private Particula[] poblacion;
    private Particula mejorGlobalSO; // gBest para SO

    // Estado Multi-Objective (MO)
    private final ArrayList<Solucion> archivo = new ArrayList<>();

    // ============================================================
    //  Estructuras de Datos Internas
    // ============================================================

    public static class Particula {
        double[] x;
        double[] v;
        
        // Memoria (PBest)
        double[] pbestPos;
        double pbestValSO;      // Para Single Objective (Scalar)
        double[] pbestObjMO;    // Para Multi Objective (Vector)

        public Particula(int dim) {
            x = new double[dim];
            v = new double[dim];
            pbestPos = new double[dim];
            // Inicializar con valores "malos"
            pbestValSO = Double.NaN; 
            pbestObjMO = new double[]{Double.MAX_VALUE, Double.MAX_VALUE};
        }
    }

    // Usado solo en MOPSO para el archivo externo
    public static class Solucion {
        double[] pos;
        double[] obj;        // [0]: Curtosis, [1]: -Sharpe
        double crowdingDist;

        public Solucion(double[] pos, double[] obj) {
            this.pos = pos;
            this.obj = obj;
            this.crowdingDist = 0.0;
        }
    }

    // ============================================================
    //  Constructor
    // ============================================================

    public PSO(double[][] retornosInput) {
        retornos = retornosInput;
        this.dimension = retornos[0].length;
    }

    // ============================================================
    //  ESTRATEGIA 1: SINGLE OBJECTIVE (Sharpe o Curtosis)
    // ============================================================

    public void maximizarSharpe(double[] rf) {
        ejecutarSingleObjective(rf, true); // true = Maximizar Sharpe
    }

    public void minimizarKurtosis() {
        ejecutarSingleObjective(null, false); // false = Minimizar Curtosis
    }

    private void ejecutarSingleObjective(double[] rf, boolean esSharpe) {
        inicializarPoblacion();
        
        // Inicializar mejor global con el peor caso posible
        mejorGlobalSO = new Particula(dimension);
        mejorGlobalSO.pbestValSO = esSharpe ? -Double.MAX_VALUE : Double.MAX_VALUE;

        // Evaluar inicial
        for (Particula p : poblacion) {
            double val = esSharpe ? calcularSharpePortafolio(p.x, rf) : calcularKurtosisPortafolio(p.x);
            
            // Actualizar PBest
            p.pbestValSO = val;
            p.pbestPos = Arrays.copyOf(p.x, dimension);

            // Actualizar GBest
            actualizarGBestSO(p, val, esSharpe);
        }

        // Ciclo principal
        for (int t = 0; t < maxIteraciones; t++) {
            for (Particula p : poblacion) {
                // 1. Actualizar velocidad y posición usando gBest como líder
                actualizarFisica(p, mejorGlobalSO.pbestPos, 0.4, 1.5, 1.5);
                
                // 2. Restricciones
                //aplicarLimites(p);
                normalizar(p.x);

                // 3. Evaluar
                double val = esSharpe ? calcularSharpePortafolio(p.x, rf) : calcularKurtosisPortafolio(p.x);

                // 4. Actualizar PBest
                boolean mejorP = esSharpe ? (val > p.pbestValSO) : (val < p.pbestValSO);
                if (mejorP) {
                    p.pbestValSO = val;
                    p.pbestPos = Arrays.copyOf(p.x, dimension);
                }

                // 5. Actualizar GBest
                actualizarGBestSO(p, val, esSharpe);
            }
        }
    }

    private void actualizarGBestSO(Particula p, double val, boolean esSharpe) {
        boolean mejorG = esSharpe ? (val > mejorGlobalSO.pbestValSO) : (val < mejorGlobalSO.pbestValSO);
        if (mejorG) {
            mejorGlobalSO.pbestValSO = val;
            mejorGlobalSO.pbestPos = Arrays.copyOf(p.x, dimension);
            // Copiamos a x también para poder recuperarlo con getMejorPosicion()
            mejorGlobalSO.x = Arrays.copyOf(p.x, dimension); 
        }
    }

    // ============================================================
    //  ESTRATEGIA 2: MULTI OBJECTIVE (MOPSO)
    // ============================================================

    public void optimizar(double[] rf) {
        inicializarPoblacion();
        archivo.clear();

        // 1. Evaluación inicial e inserción en archivo
        for (Particula p : poblacion) {
            double[] obj = evaluarMO(p.x, rf);
            actualizarPbestMO(p, obj);
            agregarAlArchivo(p.x, obj);
        }

        // Ciclo principal
        for (int t = 0; t < maxIteraciones; t++) {
            
            // Calcular distancias para selección de líder
            if (!archivo.isEmpty()) {
                calcularCrowdingDistances();
            }

            for (Particula p : poblacion) {
                // a. Selección de Líder (Torneo Crowding)
                Solucion lider = seleccionarLiderArchivo();
                double[] guia = (lider != null) ? lider.pos : p.pbestPos;

                // b. Movimiento (Inercia + Cognitivo + Social)
                // Usamos coeficientes ligeramente más altos en MOPSO para exploración
                actualizarFisica(p, guia, 0.4, 1.8, 1.8); 
                
                // c. Mutación (Turbulencia) - Exclusivo de MOPSO
                //aplicarMutacion(p, t);

                // d. Límites
                //aplicarLimites(p);
                normalizar(p.x);

                // e. Evaluación
                double[] nuevosObj = evaluarMO(p.x, rf);

                // f. Actualizar PBest y Archivo
                actualizarPbestMO(p, nuevosObj);
                agregarAlArchivo(p.x, nuevosObj);
            }
        }
    }

    // ============================================================
    //  Lógica Física Compartida (Movimiento)
    // ============================================================

    private void inicializarPoblacion() {
        poblacion = new Particula[maxPoblacion];
        for (int i = 0; i < maxPoblacion; i++) {
            poblacion[i] = new Particula(dimension);
            for (int j = 0; j < dimension; j++) {
                poblacion[i].x[j] = rnd.nextDouble();
                poblacion[i].v[j] = 0.0;
            }
            normalizar(poblacion[i].x);
        }
    }

    private void actualizarFisica(Particula p, double[] liderPos, double w, double c1, double c2) {
        for (int j = 0; j < dimension; j++) {
            double r1 = rnd.nextDouble();
            double r2 = rnd.nextDouble();
            
            // Velocidad estándar de PSO
            p.v[j] = w * p.v[j] 
                   + c1 * r1 * (p.pbestPos[j] - p.x[j]) 
                   + c2 * r2 * (liderPos[j] - p.x[j]);
            
            p.x[j] += p.v[j];
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

    private void aplicarMutacion(Particula p, int t) {
        double prob = Math.pow(1.0 - (double)t / maxIteraciones, 5.0 / tasaMutacion);
        
        if (rnd.nextDouble() < prob) {
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
    //  Lógica Específica MOPSO (Archivo & Dominancia)
    // ============================================================

    private void actualizarPbestMO(Particula p, double[] obj) {
        int d = domina(obj, p.pbestObjMO);
        if (d == -1) { // Nuevo domina a pbest
            p.pbestObjMO = obj;
            p.pbestPos = Arrays.copyOf(p.x, dimension);
        } else if (d == 0 && rnd.nextBoolean()) { // No dominados, azar
            p.pbestObjMO = obj;
            p.pbestPos = Arrays.copyOf(p.x, dimension);
        }
    }

    private double[] evaluarMO(double[] w, double[] rf) {
        double k = calcularKurtosisPortafolio(w);
        double s = calcularSharpePortafolio(w, rf);
        
        // Sanitize
        if (Double.isNaN(k) || Double.isInfinite(k)) k = 1e6; 
        if (Double.isNaN(s) || Double.isInfinite(s)) s = -1e6; 

        // Objetivos a MINIMIZAR: [Kurtosis, -Sharpe]
        return new double[]{ k, -s }; 
    }

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
                esDominado = true; 
                break;
            }
        }

        if (esDominado) return;

        archivo.removeAll(aEliminar);
        archivo.add(new Solucion(pos.clone(), obj.clone()));

        if (archivo.size() > tamanoArchivo) {
            controlarTamanoArchivo();
        }
    }

    private void controlarTamanoArchivo() {
        calcularCrowdingDistances();
        // Ordenar por distancia (menor a mayor)
        Collections.sort(archivo, Comparator.comparingDouble(s -> s.crowdingDist));
        // Eliminar el más aglomerado (menor distancia)
        archivo.remove(0);
    }

    private void calcularCrowdingDistances() {
        int n = archivo.size();
        if (n == 0) return;
        for (Solucion s : archivo) s.crowdingDist = 0.0;

        int numObj = 2; 
        for (int m = 0; m < numObj; m++) {
            final int objIndex = m;
            Collections.sort(archivo, Comparator.comparingDouble(s -> s.obj[objIndex]));

            archivo.get(0).crowdingDist = Double.POSITIVE_INFINITY;
            archivo.get(n - 1).crowdingDist = Double.POSITIVE_INFINITY;

            double minVal = archivo.get(0).obj[objIndex];
            double maxVal = archivo.get(n - 1).obj[objIndex];
            double rango = maxVal - minVal;
            if (rango == 0) rango = 1.0;

            for (int i = 1; i < n - 1; i++) {
                if (archivo.get(i).crowdingDist != Double.POSITIVE_INFINITY) {
                    double distance = (archivo.get(i + 1).obj[objIndex] - archivo.get(i - 1).obj[objIndex]) / rango;
                    archivo.get(i).crowdingDist += distance;
                }
            }
        }
    }

    private Solucion seleccionarLiderArchivo() {
        if (archivo.isEmpty()) return null;
        int idx1 = rnd.nextInt(archivo.size());
        int idx2 = rnd.nextInt(archivo.size());
        Solucion s1 = archivo.get(idx1);
        Solucion s2 = archivo.get(idx2);
        return (s1.crowdingDist > s2.crowdingDist) ? s1 : s2;
    }

    public static int domina(double[] A, double[] B) {
        // -1: A domina B, 1: B domina A, 0: No dominados
        if (A == null || B == null) return 0;
        boolean mejorEnAlgo = false;
        boolean peorEnAlgo = false;

        for (int i = 0; i < A.length; i++) {
            if (A[i] < B[i]) mejorEnAlgo = true; // Menor es mejor (Minimización)
            if (A[i] > B[i]) peorEnAlgo = true;
        }

        if (mejorEnAlgo && !peorEnAlgo) return -1; 
        if (peorEnAlgo && !mejorEnAlgo) return 1;  
        return 0;
    }

    // ============================================================
    //  Estadísticas y Métodos Públicos de Interfaz
    // ============================================================

    // Método para recuperar el mejor resultado de Single Objective
    public double[] getMejorPosicion() {
        if (mejorGlobalSO != null) {
            return Arrays.copyOf(mejorGlobalSO.x, dimension);
        }
        return new double[dimension]; // Fallback
    }

    // Métodos para recuperar resultados de Multi Objective
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

    // --- Matemáticas Financieras ---

    // Curtosis de Fisher (Normal = 0)
    public static double calcularKurtosisPortafolio(double[] w) {
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
        return (s4 / (s2 * s2)) - 3.0; // Restamos 3 para Fisher
    }

    public static double calcularSharpePortafolio(double[] w, double[] rf) {
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
}