#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <cstdint>
#include <limits>

// EwmStepper class mimicking the Python version
class EwmStepper {
public:
    // Constructor initializes folder, name, window and computes alpha
    EwmStepper(const std::string& folder = ".", const std::string& name = "default", int window = 1)
        : folder(folder), name(name), window(window)
    {
        // Compute alpha from half-life: alpha = 1 - exp(log(0.5)/window)
        alpha = 1 - std::exp(std::log(0.5) / window);
    }

    // Save internal state to a file in the given folder.
    void save() const {
        // Ensure folder exists
        std::filesystem::create_directories(folder);
        std::string filepath = folder + "/" + name + ".txt";
        std::ofstream ofs(filepath);
        if (!ofs) {
            throw std::runtime_error("Could not open file for saving state");
        }
        // Write window and alpha first
        ofs << window << "\n" << alpha << "\n";
        // Save last_sum
        ofs << last_sum.size() << "\n";
        for (const auto& kv : last_sum) {
            ofs << kv.first << " " << kv.second << "\n";
        }
        // Save last_wgt_sum
        ofs << last_wgt_sum.size() << "\n";
        for (const auto& kv : last_wgt_sum) {
            ofs << kv.first << " " << kv.second << "\n";
        }
        // Save last_timestamps
        ofs << last_timestamps.size() << "\n";
        for (const auto& kv : last_timestamps) {
            ofs << kv.first << " " << kv.second << "\n";
        }
        ofs.close();
    }

    // Load instance from saved state (or create new if file not found)
    static EwmStepper load(const std::string& folder, const std::string& name, int window = 1) {
        std::string filepath = folder + "/" + name + ".txt";
        if (!std::filesystem::exists(filepath)) {
            // File doesn't exist, return a new instance.
            return EwmStepper(folder, name, window);
        }
        std::ifstream ifs(filepath);
        if (!ifs) {
            throw std::runtime_error("Could not open state file for loading");
        }
        int fileWindow;
        double fileAlpha;
        ifs >> fileWindow >> fileAlpha;
        EwmStepper instance(folder, name, fileWindow);
        instance.alpha = fileAlpha; // set alpha from file

        size_t map_size;
        // Load last_sum map
        ifs >> map_size;
        for (size_t i = 0; i < map_size; i++) {
            int64_t key;
            double value;
            ifs >> key >> value;
            instance.last_sum[key] = value;
        }
        // Load last_wgt_sum map
        ifs >> map_size;
        for (size_t i = 0; i < map_size; i++) {
            int64_t key;
            double value;
            ifs >> key >> value;
            instance.last_wgt_sum[key] = value;
        }
        // Load last_timestamps map
        ifs >> map_size;
        for (size_t i = 0; i < map_size; i++) {
            int64_t key;
            int64_t value;
            ifs >> key >> value;
            instance.last_timestamps[key] = value;
        }
        ifs.close();
        return instance;
    }

    // Update EWM values for each code given the input vectors.
    // dt: timestamps (as int64_t, e.g. nanoseconds)
    // dscode: categorical code for each row
    // serie: values to update with
    // Returns a vector of EWM values (one per row)
    std::vector<double> update(const std::vector<int64_t>& dt,
                               const std::vector<int64_t>& dscode,
                               const std::vector<double>& serie)
    {
        if (dt.size() != dscode.size() || dt.size() != serie.size()) {
            throw std::invalid_argument("All inputs must have the same length");
        }

        std::vector<double> result(dt.size(), 0.0);
        for (size_t i = 0; i < dt.size(); i++) {
            int64_t code = dscode[i];
            double value = serie[i];
            int64_t ts = dt[i];

            // Check that timestamps for the given code are strictly increasing
            if (last_timestamps.find(code) != last_timestamps.end() && ts < last_timestamps[code]) {
                throw std::runtime_error("DateTime must be strictly increasing per code");
            }

            // If value is NaN, return NaN for this row and update the timestamp.
            if (std::isnan(value)) {
                result[i] = std::numeric_limits<double>::quiet_NaN();
                last_timestamps[code] = ts;
                continue;
            }

            // Update EWM value:
            // If this is the first value (or the stored value is NaN), initialize.
            if (last_sum.find(code) == last_sum.end() || std::isnan(last_sum[code])) {
                last_sum[code] = value;
                last_wgt_sum[code] = 1.0;
            } else {
                double old_last_sum = last_sum[code];
                double old_last_wgt_sum = last_wgt_sum[code];
                last_sum[code] = old_last_sum * (1 - alpha) + value;
                last_wgt_sum[code] = old_last_wgt_sum * (1 - alpha) + 1.0;
            }
            // Compute the current EWM value for this row.
            result[i] = last_sum[code] / last_wgt_sum[code];
            last_timestamps[code] = ts;
        }
        // Save state after update
        save();
        return result;
    }

private:
    std::string folder;
    std::string name;
    int window;
    double alpha;
    std::unordered_map<int64_t, double> last_sum;
    std::unordered_map<int64_t, double> last_wgt_sum;
    std::unordered_map<int64_t, int64_t> last_timestamps;
};

//
// Test the C++ EwmStepper in main()
//
int main() {
    try {
        // Create an instance of EwmStepper with a window of 10 (half-life)
        EwmStepper stepper("data", "ewm_state", 10);

        // Test input data:
        // dt: timestamps, dscode: categorical codes, serie: corresponding values
        std::vector<int64_t> dt = {1000, 2000, 3000, 4000, 5000};
        std::vector<int64_t> dscode = {1, 1, 2, 1, 2};
        std::vector<double> serie = {10.0, 20.0, 30.0, 40.0, 50.0};

        // Call update to process the input data and get EWM results
        std::vector<double> result = stepper.update(dt, dscode, serie);

        // Print out the results
        std::cout << "EWM Results:" << std::endl;
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << "Row " << i << ": " << result[i] << std::endl;
        }

        // Demonstrate loading state from file
        EwmStepper loaded = EwmStepper::load("data", "ewm_state");
        std::cout << "\nLoaded EwmStepper state and updating with new data..." << std::endl;

        // New test data for the loaded instance
        std::vector<int64_t> dt2 = {6000, 7000};
        std::vector<int64_t> dscode2 = {1, 2};
        std::vector<double> serie2 = {60.0, 70.0};
        std::vector<double> result2 = loaded.update(dt2, dscode2, serie2);

        // Print the new EWM results
        std::cout << "New EWM Results:" << std::endl;
        for (size_t i = 0; i < result2.size(); i++) {
            std::cout << "Row " << i << ": " << result2[i] << std::endl;
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
