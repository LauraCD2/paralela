// main.cpp
// N-body 2D (Leapfrog) + OpenMP + SFML
//
// Compilación (Linux / MinGW):
//   g++ -O3 -march=native -fopenmp main.cpp -lsfml-graphics -lsfml-window -lsfml-system -o nbody
//
// Compilación (MSVC Developer Prompt):
//   cl /O2 /openmp main.cpp /I"path\to\SFML\include" /link /LIBPATH:"path\to\SFML\lib" sfml-graphics.lib sfml-window.lib sfml-system.lib
//
// Asegúrate de tener SFML correctamente instalado/enlazado.

#include <SFML/Graphics.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
    #include <omp.h>
#else
    // Fallback si compilas sin OpenMP: el código corre igual, solo que en 1 hilo.
    inline int  omp_get_max_threads() { return 1; }
    inline void omp_set_num_threads(int) {}
#endif

// -------------------- Modelos de datos --------------------
struct Vector3
{
    double e[3] = {0.0, 0.0, 0.0};
    Vector3() = default;
    Vector3(double e0, double e1, double e2) { e[0]=e0; e[1]=e1; e[2]=e2; }
};

struct OrbitalEntity
{
    // e: [x,y,z,vx,vy,vz,m]
    double e[7] = {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0};
    Vector3 a; // aceleración
    OrbitalEntity() = default;
    OrbitalEntity(double e0,double e1,double e2,double e3,double e4,double e5,double e6)
    {
        e[0]=e0; e[1]=e1; e[2]=e2;
        e[3]=e3; e[4]=e4; e[5]=e5;
        e[6]=e6;
    }
};

enum { X=0, Y=1, Z=2, VX=3, VY=4, VZ=5, M=6 };

static inline void compute_accelerations_snapshot(
    OrbitalEntity* oe, int N,
    double* xs, double* ys, double* ms,
    const double BIG_G, const double eps2,
    const int chunk
){
    // 1) Snapshot de posiciones + masas (mejor localidad de memoria para el inner loop)
    #pragma omp for schedule(static, chunk)
    for(int i = 0; i < N; ++i)
    {
        xs[i] = oe[i].e[X];
        ys[i] = oe[i].e[Y];
        ms[i] = oe[i].e[M];
    }
    // Barrera implícita al final del omp for

    // 2) Aceleraciones
    #pragma omp for schedule(static, chunk)
    for(int i = 0; i < N; ++i)
    {
        double ax = 0.0, ay = 0.0;
        const double xi = xs[i];
        const double yi = ys[i];

        for(int j = 0; j < N; ++j)
        {
            if(i == j) continue;

            const double dx = xs[j] - xi;
            const double dy = ys[j] - yi;

            const double r2 = dx*dx + dy*dy + eps2;
            const double inv_r  = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;

            // G*mj/r^3 * (dx,dy)
            const double s = BIG_G * ms[j] * inv_r3;
            ax += s * dx;
            ay += s * dy;
        }

        oe[i].a.e[0] = ax;
        oe[i].a.e[1] = ay;
        oe[i].a.e[2] = 0.0;
    }
}

static inline double rand_uniform(std::mt19937_64& rng, double a, double b)
{
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

static inline double rand_log10_uniform(std::mt19937_64& rng, double log10_a, double log10_b)
{
    std::uniform_real_distribution<double> dist(log10_a, log10_b);
    return std::pow(10.0, dist(rng));
}

static void init_asteroids(
    std::vector<OrbitalEntity>& oe,
    int start_index,
    int count,
    double BIG_G,
    double AU
){
    if(count <= 0) return;

    constexpr double PI = 3.1415926535897932384626433832795;

    std::mt19937_64 rng(123456789ULL);

    const double M_sun = oe[0].e[M];

    // Cinturón tipo "asteroid belt" (aprox. 2.0 a 3.5 AU)
    const double r_min = 2.0 * AU;
    const double r_max = 3.5 * AU;

    for(int k = 0; k < count; ++k)
    {
        const double theta = rand_uniform(rng, 0.0, 2.0 * PI);
        const double r     = rand_uniform(rng, r_min, r_max);

        const double x = r * std::cos(theta);
        const double y = r * std::sin(theta);

        // Velocidad circular alrededor del Sol (aprox)
        const double v_circ = std::sqrt(BIG_G * M_sun / r);

        // Dirección tangencial (perpendicular al radio)
        double vx = -v_circ * std::sin(theta);
        double vy =  v_circ * std::cos(theta);

        // Pequeña perturbación (para que no sea todo perfecto)
        vx *= (1.0 + rand_uniform(rng, -0.02, 0.02));
        vy *= (1.0 + rand_uniform(rng, -0.02, 0.02));

        // Masa del asteroide (log-uniform entre 1e12 y 1e18 kg, ajusta si quieres)
        const double mass = rand_log10_uniform(rng, 12.0, 18.0);

        const int idx = start_index + k;
        oe[idx] = OrbitalEntity(x, y, 0.0, vx, vy, 0.0, mass);
    }
}

int main()
{
    // -------------------- Parámetros físicos y numéricos --------------------
    const double BIG_G = 6.67430e-11;
    const double dt    = 86400.0;        // 1 día
    const double years = 50.0;           // Para pruebas rápidas, baja a 1-5
    const double AU    = 1.496e11;

    // Softening: evita singularidad si dos cuerpos se acercan demasiado.
    // (si eps es muy pequeño, pueden aparecer aceleraciones enormes y explotar)
    const double eps  = 1e6;             // 1000 km
    const double eps2 = eps * eps;

    // Guardar trayectoria cada k pasos (reduce memoria y acelera visualización)
    const int store_every = 1;           // prueba 2, 5, 10 si te pesa

    // -------------------- Cuerpos --------------------
    // 9 = Sol + 8 planetas del ejemplo
    // Para total ~1000: N_ASTEROIDS = 991
    const int N_ASTEROIDS = 991;
    const int N_MAJOR     = 9;           // Sol + 8 planetas
    const int N           = N_MAJOR + N_ASTEROIDS;

    std::vector<OrbitalEntity> orbital_entities(N);

    // Sol + 8 planetas (2D simplificado)
    orbital_entities[0] = {0,0,0, 0,0,0, 1.989e30};      // Sol
    orbital_entities[1] = {57.909e9,0,0, 0,47.36e3,0, 0.33011e24};  // Mercurio
    orbital_entities[2] = {108.209e9,0,0, 0,35.02e3,0, 4.8675e24};  // Venus
    orbital_entities[3] = {149.596e9,0,0, 0,29.78e3,0, 5.9724e24};  // Tierra
    orbital_entities[4] = {227.923e9,0,0, 0,24.07e3,0, 0.64171e24}; // Marte
    orbital_entities[5] = {778.570e9,0,0, 0,13.00e3,0, 1898.19e24}; // Júpiter
    orbital_entities[6] = {1433.529e9,0,0, 0,9.68e3,0, 568.34e24};  // Saturno
    orbital_entities[7] = {2872.463e9,0,0, 0,6.80e3,0, 86.813e24};  // Urano
    orbital_entities[8] = {4495.060e9,0,0, 0,5.43e3,0, 102.413e24}; // Neptuno

    // Asteroides generados automáticamente
    init_asteroids(orbital_entities, N_MAJOR, N_ASTEROIDS, BIG_G, AU);

    const int steps_total = static_cast<int>((years * 365.0 * dt) / dt); // years*365
    const int saved_steps = steps_total / store_every + 1;

    // Trayectorias
    std::vector<std::vector<sf::Vector2f>> trajectories(N);
    for(auto& tr : trajectories) tr.reserve(saved_steps);

    // Buffers snapshot (SoA)
    std::vector<double> xs(N), ys(N), ms(N);

    // OpenMP threads
    omp_set_num_threads(16); // tu 5700X: 16 hilos lógicos (ajusta si quieres)
    const int T = omp_get_max_threads();
    const int chunk = std::max(1, N / (T * 2)); // "bloques" de i (prueba 16/32/64 también)

#ifdef _OPENMP
    std::cout << "OpenMP ENABLED | threads=" << T << " | chunk=" << chunk << "\n";
#else
    std::cout << "OpenMP NOT enabled (compila con OpenMP para paralelizar)\n";
#endif
    std::cout << "N=" << N << " (major=" << N_MAJOR << ", asteroids=" << N_ASTEROIDS << ")\n";
    std::cout << "years=" << years << " dt=" << dt << " steps=" << steps_total
              << " store_every=" << store_every << " saved_steps~" << saved_steps << "\n";

    // -------------------- Simulación --------------------
    auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        // a(t=0)
        compute_accelerations_snapshot(
            orbital_entities.data(), N,
            xs.data(), ys.data(), ms.data(),
            BIG_G, eps2, chunk
        );

        // Guardar estado inicial (t=0)
        #pragma omp for schedule(static, chunk)
        for(int i = 0; i < N; ++i)
        {
            trajectories[i].push_back(sf::Vector2f(
                static_cast<float>(orbital_entities[i].e[X] / AU),
                static_cast<float>(orbital_entities[i].e[Y] / AU)
            ));
        }

        for(int step = 1; step <= steps_total; ++step)
        {
            // v(t+dt/2)
            #pragma omp for schedule(static, chunk)
            for(int i = 0; i < N; ++i)
            {
                orbital_entities[i].e[VX] += 0.5 * orbital_entities[i].a.e[0] * dt;
                orbital_entities[i].e[VY] += 0.5 * orbital_entities[i].a.e[1] * dt;
            }

            // x(t+dt)
            #pragma omp for schedule(static, chunk)
            for(int i = 0; i < N; ++i)
            {
                orbital_entities[i].e[X] += orbital_entities[i].e[VX] * dt;
                orbital_entities[i].e[Y] += orbital_entities[i].e[VY] * dt;
            }

            // a(t+dt)
            compute_accelerations_snapshot(
                orbital_entities.data(), N,
                xs.data(), ys.data(), ms.data(),
                BIG_G, eps2, chunk
            );

            // v(t+dt) + guardar trayectoria (cada store_every)
            #pragma omp for schedule(static, chunk)
            for(int i = 0; i < N; ++i)
            {
                orbital_entities[i].e[VX] += 0.5 * orbital_entities[i].a.e[0] * dt;
                orbital_entities[i].e[VY] += 0.5 * orbital_entities[i].a.e[1] * dt;

                if(step % store_every == 0)
                {
                    trajectories[i].push_back(sf::Vector2f(
                        static_cast<float>(orbital_entities[i].e[X] / AU),
                        static_cast<float>(orbital_entities[i].e[Y] / AU)
                    ));
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    const double sim_seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Simulacion terminada. Puntos guardados: " << trajectories[0].size()
              << " | tiempo: " << sim_seconds << " s\n";

    // -------------------- VISUALIZACIÓN (SFML) --------------------
    sf::RenderWindow window(sf::VideoMode(1200, 1200), "N-Body 2D (Leapfrog + OpenMP)");
    window.setFramerateLimit(60);

    // Vista inicial (en AU). Ajusta como quieras.
    sf::View view(sf::FloatRect(-8.f, -8.f, 16.f, 16.f));
    window.setView(view);

    int current_step = 1;
    const int max_steps = static_cast<int>(trajectories[0].size());
    int speed = 1;

    // Para no matar la GPU/CPU: dibuja órbitas solo para cuerpos mayores
    const bool draw_major_orbits = true;
    const int  orbit_tail = 3000; // cuántos puntos de "cola" dibujar (reduce carga)

    // Cuerpos (círculos) para mayores
    sf::CircleShape body;
    body.setFillColor(sf::Color::Green);

    while(window.isOpen())
    {
        sf::Event event;
        while(window.pollEvent(event))
        {
            if(event.type == sf::Event::Closed)
                window.close();

            if(event.type == sf::Event::MouseWheelScrolled)
            {
                if(event.mouseWheelScroll.delta > 0) view.zoom(0.9f);
                else view.zoom(1.1f);
                window.setView(view);
            }

            if(event.type == sf::Event::KeyPressed)
            {
                if(event.key.code == sf::Keyboard::Add) speed++;
                if(event.key.code == sf::Keyboard::Subtract && speed > 1) speed--;

                // Flechas para panear (opcional)
                if(event.key.code == sf::Keyboard::Left)  view.move(-0.5f, 0.f);
                if(event.key.code == sf::Keyboard::Right) view.move( 0.5f, 0.f);
                if(event.key.code == sf::Keyboard::Up)    view.move(0.f, -0.5f);
                if(event.key.code == sf::Keyboard::Down)  view.move(0.f,  0.5f);
                window.setView(view);
            }
        }

        window.clear(sf::Color::Black);

        // Dibujar órbitas de los cuerpos mayores (Sol+planetas)
        if(draw_major_orbits && current_step > 2)
        {
            const int major_count = std::min(N_MAJOR, N);
            for(int i = 0; i < major_count; ++i)
            {
                const int end   = std::min(current_step, max_steps);
                const int start = std::max(0, end - orbit_tail);
                const int count = end - start;

                if(count < 2) continue;

                sf::VertexArray orbit(sf::LineStrip, static_cast<std::size_t>(count));
                for(int k = 0; k < count; ++k)
                {
                    orbit[k].position = trajectories[i][start + k];
                    orbit[k].color    = sf::Color::White;
                }
                window.draw(orbit);
            }
        }

        // Dibujar posiciones actuales: mayores como círculos
        if(current_step > 0)
        {
            const int idx = std::min(current_step - 1, max_steps - 1);

            for(int i = 0; i < std::min(N_MAJOR, N); ++i)
            {
                float radius = 0.08f;
                sf::Color c = sf::Color::Green;

                if(i == 0) { radius = 0.18f; c = sf::Color(255, 220, 0); } // Sol

                body.setRadius(radius);
                body.setOrigin(radius, radius);
                body.setFillColor(c);
                body.setPosition(trajectories[i][idx]);
                window.draw(body);
            }

            // Asteroides como puntos (mucho más barato que círculos u órbitas)
            if(N > N_MAJOR)
            {
                sf::VertexArray points(sf::Points, static_cast<std::size_t>(N - N_MAJOR));
                for(int i = N_MAJOR; i < N; ++i)
                {
                    const std::size_t p = static_cast<std::size_t>(i - N_MAJOR);
                    points[p].position = trajectories[i][idx];
                    points[p].color    = sf::Color(180, 180, 180);
                }
                window.draw(points);
            }
        }

        if(current_step < max_steps) current_step = std::min(max_steps, current_step + speed);

        window.display();
    }

    return 0;
}