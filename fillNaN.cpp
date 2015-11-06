#include <cmath>
// #include <cstdlib>
#include <limits>
#include <random>

namespace {
std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> dist(0, 1);
const double NaN = std::numeric_limits<double>::quiet_NaN();
}

double generateValue()
{
    return dist(e2);
}

double* createHistory(int numValues)
{
    auto values = reinterpret_cast<double *>(
            std::malloc(sizeof(double) * numValues));
    
    for (int i = 0; i < numValues; ++i)
    {
        if (i % 7 > 5
            || generateValue() <= (10.0 / 365.0))
        {
            values[i] = NaN;
        }
        else
        {
            values[i] = generateValue();
        }
    }
    
    return values;
}

double* cloneHistory(const double* values, int numValues)
{
    auto newValues = reinterpret_cast<double *>(
            std::malloc(sizeof(double) * numValues));

    for (int i = 0; i < numValues; ++i)
    {
        newValues[i] = values[i];
    }
    
    return newValues;
}

void destroyHistory(double* values)
{
    std::free(values);
}

void fillNaN(double* values, int numValues)
{
    if (numValues == 0)
    {
        return;
    }

    auto lastValid = values[0];
    for (int i = 1; i < numValues; ++i)
    {
        double& value = values[i];
        if (std::isnan(value))
        {
            value = lastValid;
        }
        else
        {
            lastValid = value;
        }
    }
}

void fillNaN_reversed(double* values, int numValues)
{
    if (numValues == 0)
    {
        return;
    }
    
    auto lastValid = values[numValues - 1];
    for (int i = numValues - 1; i >= 0; --i)
    {
        double& value = values[i];
        if (std::isnan(value))
        {
            value = lastValid;
        }
        else
        {
            lastValid = value;
        }
    }
}