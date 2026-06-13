import com.aparapi.Kernel;
import com.aparapi.Range;

import java.util.Random;

public class NBodySimulation {

    // ---- Aparapi Kernel: paralel calisan kisim ----
    static class NBodyKernel extends Kernel {
        final float[] posX, posY, posZ;
        final float[] velX, velY, velZ;
        final float[] mass;
        final int N;
        final float G = 6.67430e-11f;
        final float DT = 0.01f;

        NBodyKernel(float[] posX, float[] posY, float[] posZ,
                     float[] velX, float[] velY, float[] velZ,
                     float[] mass, int N) {
            this.posX = posX; this.posY = posY; this.posZ = posZ;
            this.velX = velX; this.velY = velY; this.velZ = velZ;
            this.mass = mass; this.N = N;
        }

        @Override
        public void run() {
            int i = getGlobalId();

            float xi = posX[i], yi = posY[i], zi = posZ[i];
            float fx = 0f, fy = 0f, fz = 0f;

            for (int j = 0; j < N; j++) {
                if (j != i) {
                    float dx = posX[j] - xi;
                    float dy = posY[j] - yi;
                    float dz = posZ[j] - zi;
                    float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f;
                    float invDist = 1.0f / sqrt(distSqr);
                    float invDist3 = invDist * invDist * invDist;
                    float f = G * mass[j] * invDist3;

                    fx += f * dx;
                    fy += f * dy;
                    fz += f * dz;
                }
            }

            velX[i] += fx * DT;
            velY[i] += fy * DT;
            velZ[i] += fz * DT;

            posX[i] += velX[i] * DT;
            posY[i] += velY[i] * DT;
            posZ[i] += velZ[i] * DT;
        }
    }

    // ---- Seri (CPU) versiyon: karsilastirma icin ----
    static void runSerial(float[] posX, float[] posY, float[] posZ,
                           float[] velX, float[] velY, float[] velZ,
                           float[] mass, int N, int steps) {
        final float G = 6.67430e-11f;
        final float DT = 0.01f;

        for (int s = 0; s < steps; s++) {
            for (int i = 0; i < N; i++) {
                float xi = posX[i], yi = posY[i], zi = posZ[i];
                float fx = 0f, fy = 0f, fz = 0f;

                for (int j = 0; j < N; j++) {
                    if (j != i) {
                        float dx = posX[j] - xi;
                        float dy = posY[j] - yi;
                        float dz = posZ[j] - zi;
                        float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f;
                        float invDist = (float) (1.0 / Math.sqrt(distSqr));
                        float invDist3 = invDist * invDist * invDist;
                        float f = G * mass[j] * invDist3;

                        fx += f * dx;
                        fy += f * dy;
                        fz += f * dz;
                    }
                }

                velX[i] += fx * DT;
                velY[i] += fy * DT;
                velZ[i] += fz * DT;
            }

            // pozisyon guncelleme (ayri dongude, kernel ile ayni sirayi tutmak icin)
            for (int i = 0; i < N; i++) {
                posX[i] += velX[i] * DT;
                posY[i] += velY[i] * DT;
                posZ[i] += velZ[i] * DT;
            }
        }
    }

    public static void main(String[] args) {
        int N = (args.length > 0) ? Integer.parseInt(args[0]) : 1000;
        int STEPS = (args.length > 1) ? Integer.parseInt(args[1]) : 50;

        System.out.println("N = " + N + ", STEPS = " + STEPS);

        // ---- Veri dizilerini hazirla (paralel icin) ----
        float[] posX = new float[N], posY = new float[N], posZ = new float[N];
        float[] velX = new float[N], velY = new float[N], velZ = new float[N];
        float[] mass = new float[N];

        Random rnd = new Random(42);
        for (int i = 0; i < N; i++) {
            posX[i] = rnd.nextFloat() * 1000f;
            posY[i] = rnd.nextFloat() * 1000f;
            posZ[i] = rnd.nextFloat() * 1000f;
            velX[i] = 0f; velY[i] = 0f; velZ[i] = 0f;
            mass[i] = 1f + rnd.nextFloat() * 1000f;
        }

        // ---- Seri icin ayni veriyi kopyala ----
        float[] sPosX = posX.clone(), sPosY = posY.clone(), sPosZ = posZ.clone();
        float[] sVelX = velX.clone(), sVelY = velY.clone(), sVelZ = velZ.clone();
        float[] sMass = mass.clone();

        // ---- SERI CALISTIRMA ----
        long t1 = System.nanoTime();
        runSerial(sPosX, sPosY, sPosZ, sVelX, sVelY, sVelZ, sMass, N, STEPS);
        long t2 = System.nanoTime();
        double serialMs = (t2 - t1) / 1e6;

        // ---- PARALEL (APARAPI) CALISTIRMA ----
        NBodyKernel kernel = new NBodyKernel(posX, posY, posZ, velX, velY, velZ, mass, N);
        kernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);

        Range range = Range.create(N);

        long t3 = System.nanoTime();
        for (int s = 0; s < STEPS; s++) {
            kernel.execute(range);
        }
        long t4 = System.nanoTime();
        double parallelMs = (t4 - t3) / 1e6;

        System.out.println("Calisma modu : " + kernel.getExecutionMode());
        System.out.printf("Seri sure    : %.2f ms%n", serialMs);
        System.out.printf("Paralel sure : %.2f ms%n", parallelMs);
        System.out.printf("Hizlanma     : %.2fx%n", serialMs / parallelMs);

        // ---- Dogrulama: ilk parcacigin son konumunu karsilastir ----
        System.out.printf("Seri  posX[0]=%.5f  Paralel posX[0]=%.5f%n", sPosX[0], posX[0]);

        kernel.dispose();
    }
}
