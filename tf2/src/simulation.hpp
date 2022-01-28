#include <iostream>
#include <vector>
#include <regex>
#include <fstream>
#include <random>

enum Dim {
    N, C, M, P, Q, R, S, DIMENSIONS
};

enum Datatype {
    WEIGHTS, INPUTS, OUTPUTS, DATATYPES
};

struct Loop {
    Dim dim;
    size_t level = 0;
    int bound = 0;
    bool spatial = false;
    bool deepest = false;
    int cur = 0;

    Loop(Dim d, size_t l, int b, bool s, bool f) : 
        dim(d), level(l), bound(b), spatial(s), deepest(f), cur(0) {}
};

using Errors = std::vector<uint8_t>;

struct Buffer {    
    std::array<Errors, DATATYPES> writeError;
    std::array<Errors, DATATYPES> readError;
    std::array<bool, DATATYPES> containsDatatype;

    bool approx;
    float readBer, writeBer;
    
    int level;
};

using Mapping = std::vector<Loop>;

class Simulation {
    public:
    
    Mapping mapping;
    std::vector<Buffer> buffers;

    std::vector<int> dimIdx;

    Simulation(std::string mapping_file)
    {
        std::ifstream f(mapping_file);
        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string data = buffer.str();
        
        std::regex levelRegex(R"##(([a-zA-Z0-9-_]+\b) \[ (?:Weights:([0-9]+) )?(?:Inputs:([0-9]+) )?(?:Outputs:([0-9]+) )?\] \n-+\n\n((?:\|? +for.*\n)+))##");
        std::regex loopRegex(R"##((?:\| +for ([a-z])[0-9](s)? in \[0:([0-9]+)\)))##");

        std::vector<bool> deepestForDim(DIMENSIONS, true);

        int l = 0;
        for (std::smatch bm; std::regex_search(data, bm, levelRegex); data = bm.suffix())
        {   
            std::string loopNest = bm[5];
            for (std::smatch lm; std::regex_search(loopNest, lm, loopRegex); loopNest = lm.suffix()) {
                std::string dim = lm[1];
                std::string spatial = lm[2];
                std::string bound = lm[3];

                Dim d = IdxToDim(dim);
                mapping.emplace(mapping.begin(), d, l, std::stoi(bound), spatial == "s", false);
            }

            Buffer buf;
            for (int i = 0; i < 3; i++) 
                buf.containsDatatype[dt] = bm[2+1].str().length() != 0;
            buf.approx = true;
            buf.readBer = buf.writeBer = 1e-6;
            buffers.insert(buffers.begin())

            l++;
        }

        for (int i = 0; i < buffers.size(); i++)
            buffers[i].level = i;
        
        Mapping mcopy(mapping);
        for (int d = 0; d < Dim::DIMENSIONS; d++) {
            bool found = false;
            for (auto& loop : mcopy)
                if (loop.dim == (Dim)d)
                    found = true;
            if (!found)
                mapping.emplace_back((Dim)d, mapping.back().level, 1, false, true);
        }

        for (int i = 0; i < mapping.size(); i++)
        {   
            Dim d = mapping[i].dim;
            mapping[i].deepest = deepestForDim[d];
            deepestForDim[d] = false;

            const std::string lookup = "NCMPQRS";
            //std::cout << "Dim: " << lookup[mapping[i].dim] << " - Bound: " << mapping[i].bound << " - Deepest: " << mapping[i].deepest << std::endl;
        }

        dimIdx = std::vector<int>(Dim::DIMENSIONS, 0);
    }

    void UpdateDimensionIndexes() {
        for (int d = 0; d < Dim::DIMENSIONS; d++) {
            dimIdx[d] = 0;
            int tb = 1;
            for (int i = 0; i < mapping.size(); i++)
            {
                if (mapping[i].dim == d)
                {
                    dimIdx[d] += tb * mapping[i].cur;
                    tb *= mapping[i].bound;
                }
            }
        }
    }

    bool Next() {
        mapping[0].cur++;
        UpdateDimIndex(mapping[0].dim);

        int depth = 0;
        while (mapping[depth].cur == mapping[depth].bound)
        {
            if (depth == mapping.size() - 1) {
                return true;
            }
            
            mapping[depth].cur = 0;
            UpdateDimIndex(mapping[depth].dim);

            depth++;
            mapping[depth].cur++;
            UpdateDimIndex(mapping[depth].dim);
        }

        return false;
    }

    void UpdateDimIndex(Dim d) {
        dimIdx[d] = 0;
        int tb = 1;
        for (int i = 0; i < mapping.size(); i++)
        {
            if (mapping[i].dim == d)
            {
                dimIdx[d] += tb * mapping[i].cur;
                tb *= mapping[i].bound;
            }
        }
    } 

    static bool IsDimRelevantToDatatype(Dim dim, Datatype dt) {
        const static bool relevanceMap[Datatype::DATATYPES][Dim::DIMENSIONS] = {
        /*              N       C       M       P       Q       R       S       */
        /* Weights */  {false,  true,   true,   false,  false,  true,   true},
        /* Inputs  */  {true,   true,   false,  true,   true,   true,   true},
        /* Outputs */  {false,  false,  true,   true,   true,   false,  false},
        };

        return relevanceMap[dt][dim];
    }

    private:


    Dim IdxToDim(std::string idx) {
        const std::string lookup = "ncmpqrs";
        return (Dim) lookup.find(idx);
    }

};


/*
const size_t buffer_name_g = 1;
const size_t weight_size_g = 2;
const size_t input_size_g = 3; 
const size_t output_size_g = 4; 
*/

/*

void addLoops(Mapping& mapping, int b, Dimension d)
{      
    if (b == 1) {
        mapping.emplace(mapping.begin(), d, 0, 1, false, false);
        return;
    }

    for (int i = 2; i < b; i++)
        if (b % i == 0)
        {
            mapping.emplace(mapping.begin(), d, 0, i, false, false);
            mapping.emplace(mapping.begin(), d, 0, b / i, false, false);
            break;
        }
}

Mapping TestMapping(int n, int c, int m, int p, int q, int r, int s)
{
    Mapping mapping;

    addLoops(mapping, n, N);
    addLoops(mapping, c, C);
    addLoops(mapping, m, M);
    addLoops(mapping, p, P);
    addLoops(mapping, q, Q);
    addLoops(mapping, r, R);
    addLoops(mapping, s, S);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(mapping), std::end(mapping), rng);

    std::vector<bool> deepestForDim(DIMENSIONS, true);

    std::cout << "LOOP_NEST" << std::endl;
    for (int i = 0; i < mapping.size(); i++)
    {   
        Dimension d = mapping[i].dim;
        mapping[i].deepest = deepestForDim[d];
        deepestForDim[d] = false;

        const std::string lookup = "NCMPQRS";
        std::cout << "Dim: " << lookup[d] << " - Bound: " << mapping[i].bound << " - Deepest: " << mapping[i].deepest << std::endl;

    }

    return mapping;
}
*/