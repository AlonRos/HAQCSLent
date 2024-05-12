#include "Shor'sAlgorithm.h"

int f(int g, int x, int N)
{
    int res = 1;     // Initialize result 

    g = g % N; // Update x if it is more than or 
    // equal to p

    if (g == 0) return 0; // In case x is divisible by p;

    while (x > 0)
    {
        // If y is odd, multiply x with result 
        if (x & 1)
            res = (res * g) % N;

        // y must be even now 
        x = x >> 1; // y = y/2 
        g = (g * g) % N;
    }
    return res;
}

void factor(int N) {
    int n = (int)ceil(log2(N));
    int Q = 1 << (2 * n + 1);

    int g = 7;//randBound(N - 1) + 1;
    cout << g << "\n\n";

    Quregister q(2 * n + 1 + n, 0);

    q.getCoords()->entry(0, 0) = 0;

    for (int x = 0; x < Q; ++x) {
        int l = f(g, x, N);
        q.getCoords()->entry((x << n) | f(g, x, N), 0) = 1 / sqrt(Q);
    }

    cout << q.regMeasureComputational(0, n) << "\n\n";

    //q.getCoords()->print();

}