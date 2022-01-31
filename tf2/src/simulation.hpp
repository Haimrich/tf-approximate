#include <iostream>
#include <vector>
#include <regex>
#include <fstream>
#include <random>
#include <limits>
#include <bitset>

enum Dim {
    N, C, M, P, Q, R, S, DIMENSIONS
};

enum Datatype {
    WEIGHTS, INPUTS, OUTPUTS, DATATYPES
};

struct Buffer {
    std::string name;
    Datatype datatype;

    std::vector<uint8_t> writeError;
    std::vector<uint8_t> readError;

    bool approximatedRead, approximatedWrite;
    float readBer, writeBer;
    
    // Mapping loop index start and end
    int mlStart, mlEnd;
    int level;

    std::mt19937 random;

    Buffer(std::string name, Datatype dt, int l) :
        name(name),
        datatype(dt),
        approximatedRead(false), 
        approximatedWrite(false), 
        readBer(0.0), writeBer(0.0),
        mlStart(0), mlEnd(0), level(l) {
            std::random_device rd;
            random = std::mt19937(rd());
        }
};


struct Loop {
    Dim dim;
    size_t level = 0;
    int bound = 0;
    bool spatial = false;
    bool deepest = false;
    int cur = 0;

    std::vector<std::function<void()>> updateCallbacks;

    Loop(Dim d, size_t l, int b, bool s, bool f) : 
        dim(d), level(l), bound(b), spatial(s), deepest(f), cur(0) {}
};

using Mapping = std::vector<Loop>;

class Simulation {
    public:
    
    Mapping mapping;
    std::array<std::vector<Buffer>, DATATYPES> buffers;

    std::vector<int> dimIdx;

    Simulation(std::string mapping_file, std::string bers_file)
    {
        std::ifstream file(mapping_file);
        std::stringstream strBuffer;
        strBuffer << file.rdbuf();
        std::string data = strBuffer.str();
        
        std::regex levelRegex(R"##(([a-zA-Z0-9-_]+\b) \[ (?:Weights:([0-9]+) )?(?:Inputs:([0-9]+) )?(?:Outputs:([0-9]+) )?\] \n-+\n\n((?:\|? +for.*\n)+))##");
        std::regex loopRegex(R"##((?:\| +for ([a-z])[0-9](?:s)? in \[0:([0-9]+)\))( \(S)?)##");

        std::vector<bool> deepestForDim(DIMENSIONS, true);

        int l = 0;
        for (std::smatch bm; std::regex_search(data, bm, levelRegex); data = bm.suffix())
        {   
            std::string loopNest = bm[5];
            for (std::smatch lm; std::regex_search(loopNest, lm, loopRegex); loopNest = lm.suffix()) {
                std::string dim = lm[1];
                std::string bound = lm[2];
                std::string spatial = lm[3];

                Dim d = IdxToDim(dim);
                mapping.emplace(mapping.begin(), d, l, std::stoi(bound), spatial.length() != 0, false);
            }

            for (int i = 0; i < DATATYPES; i++) {
                if (bm[2+i].str().length() != 0) {
                    Datatype dt = static_cast<Datatype>(i);
                    Buffer buf(bm[1], dt, l);
                    buffers[dt].insert(buffers[dt].begin(), buf);
                }
            }
            l++;
        }
        
        // Reverse levels
        for (int i = 0; i < mapping.size(); i++)
            mapping[i].level = l - mapping[i].level - 1;

        for (int i = 0; i < DATATYPES; i++)
            for (auto& buffer : buffers[i])
                buffer.level = l - buffer.level - 1;


        // Configurazione approssimazione buffers
        std::ifstream bersFile(bers_file);
        std::stringstream bersFileBuffer;
        bersFileBuffer << bersFile.rdbuf();
        std::string bersData = bersFileBuffer.str();
        
        std::regex berRegex(R"##(([a-zA-Z0-9_-]*\b) +(?:R:([0-9e.-]+))? +(?:W:([0-9e.-]+))?)##");

        for (std::smatch bm; std::regex_search(bersData, bm, berRegex); bersData = bm.suffix())
        {   
            
            std::string bufferName = bm[1];
            std::string readBerStr = bm[2];
            std::string writeBerStr = bm[3];

            for (int i = 0; i < DATATYPES; i++) {
                for (auto& buffer : buffers[i]) {
                    if (buffer.name == bufferName) {
                        if (readBerStr.length() > 0) {
                            buffer.approximatedRead = true;
                            buffer.readBer = stod(readBerStr);
                        }
                        if (writeBerStr.length() > 0) {
                            buffer.approximatedWrite = true;
                            buffer.writeBer = stod(writeBerStr);
                        }
                    }
                }
            }
        }

        // Imposta start ed end loop per i buffer
        for (int i = 0; i < DATATYPES; i++) {

            int buffer = 0;
            for (int j = 0; j < mapping.size(); j++) {

                if (mapping[j].level <= buffers[i][buffer].level) {
                    buffers[i][buffer].mlEnd = j + 1;
                } else {
                    buffer++;
                    if (buffer < buffers[i].size()) {
                        buffers[i][buffer].mlStart = j;
                        buffers[i][buffer].mlEnd = j + 1;
                    }
                }
            }
        }

        for (int i = 0; i < DATATYPES; i++) {
            std::cout << "DATATYPE:  " << i << std::endl;
            for (auto&  b : buffers[i]) {
                std::cout << b.name << " " << b.mlStart << " " << b.mlEnd << std::endl;
            }
        }
        
        // Error dims
        for (int i = 0; i < DATATYPES; i++)
            for (int buffer = 0; buffer < buffers[i].size(); buffer++) {    
                if (buffers[i][buffer].approximatedWrite) {
                    size_t writeErrorSize = std::accumulate(mapping.begin(), mapping.begin()+buffers[i][buffer].mlEnd, 1, 
                        [&](int prod, const Loop& loop) { 
                            if (IsDimRelevantToDatatype(loop.dim, buffers[i][buffer].datatype))
                                return prod * loop.bound;
                            
                            return prod;
                        });
                    std::cout << "WRITESIZE: " << writeErrorSize << std::endl;
                    buffers[i][buffer].writeError.resize(writeErrorSize);
                }
                
                if (buffers[i][buffer].approximatedRead) {
                    // TODO Questa cosa è un'approssimazione perchè per gli input non vale come al solito
                    size_t readErrorSize = 1;
                    for (int j = 0; j < buffers[i][buffer].mlEnd; j++) {
                        if (j < buffers[i][buffer].mlStart) {
                            if (IsDimRelevantToDatatype(mapping[j].dim, buffers[i][buffer].datatype))
                                readErrorSize *= mapping[j].bound;
                        } else if (!mapping[j].spatial || (mapping[j].spatial && IsDimRelevantToDatatype(mapping[j].dim, buffers[i][buffer].datatype))) {
                            readErrorSize *= mapping[j].bound;
                        }
                    }
                    /*
                    size_t readErrorSize = std::accumulate(mapping.begin(), mapping.begin()+buffers[i][buffer].mlEnd, 1, 
                        [&](int prod, const Loop& loop) { 
                            // Calcolo dimensione
                            if (loop.level < buffers[i][buffer].level) {
                                if (IsDimRelevantToDatatype(loop.dim, buffers[i][buffer].datatype))
                                    return prod * loop.bound;

                                return prod;
                            }
                            // Calcolo change cycles
                            if (!loop.spatial || (loop.spatial && IsDimRelevantToDatatype(loop.dim, buffers[i][buffer].datatype)))
                                return prod * loop.bound;
                            
                            return prod;
                        }); */
                    buffers[i][buffer].readError.resize(readErrorSize);
                }
            }

        // Generazione errori
        for (int i = 0; i < DATATYPES; i++)
            for (auto& buf : buffers[i]) {
                if (buf.approximatedRead) 
                    UpdateReadErrors(buf);
                if (buf.approximatedWrite)
                    UpdateWriteErrors(buf);

                std::cout << "Buffer: " << buf.name << std::endl;
                std::cout << " AXR: " << buf.approximatedRead << " - Size: " << buf.readError.size() << std::endl;
                std::cout << " AXW: " << buf.approximatedWrite << " - Size: " << buf.writeError.size() << std::endl;

            }

        // Setta update callbacks
        for (int i = 0; i < DATATYPES; i++) {
            for (int j = 1; j < buffers[i].size(); j++) {

                // Read errors - primo loop superiore
                std::function<void()> updateReadErrorsCallback = [&,i,j]() { UpdateReadErrors(buffers[i][j-1]); };
                mapping[buffers[i][j].mlStart].updateCallbacks.push_back(updateReadErrorsCallback);

                // Write errors - primo loop superiore (spaziale) o (temporale e relativo a datatype)
                for (int k = buffers[i][j].mlStart; k < buffers[i][j].mlEnd; k++) {
                    Datatype dt = static_cast<Datatype>(i);
                    if (mapping[k].spatial || IsDimRelevantToDatatype(mapping[k].dim, dt)) {
                        std::function<void()> updateWriteErrorsCallback = [&,i,j]() { UpdateWriteErrors(buffers[i][j-1]); };
                        mapping[k].updateCallbacks.push_back(updateWriteErrorsCallback);
                        break;
                    }
                }
            }
        }

        // Vettore degli indici cumulati
        dimIdx = std::vector<int>(Dim::DIMENSIONS, 0);
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
            UpdateErrors(depth);
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

    
    uint8_t GetError(Datatype dt) {
        uint8_t error = 0;
        for (auto& buf : buffers[dt]) {

            if (buf.approximatedWrite) {
                size_t writeIdx = 0;
                int tb = 1;
                for (int i = 0; i < buf.mlEnd; i++)
                {
                    if (IsDimRelevantToDatatype(mapping[i].dim, dt))
                    {
                        writeIdx += tb * mapping[i].cur;
                        tb *= mapping[i].bound;
                    }
                }
                error ^= buf.writeError[writeIdx];
            }

            if (buf.approximatedRead) {
                size_t readIdx = 0;
                int tb = 1;
                for (int i = 0; i < buf.mlEnd; i++)
                {
                    if (i < buf.mlStart) {
                        if (IsDimRelevantToDatatype(mapping[i].dim, dt))
                        {
                            readIdx += tb * mapping[i].cur;
                            tb *= mapping[i].bound;
                        }
                        continue;
                    } else if ( !mapping[i].spatial || (mapping[i].spatial && IsDimRelevantToDatatype(mapping[i].dim, dt)) ) {
                        readIdx += tb * mapping[i].cur;
                        tb *= mapping[i].bound;
                    }
                }
                //std::cout << "Idx: " << readIdx << " - Size: " << buf.readError.size() << std::endl;
                error ^= buf.readError[readIdx];
            }

            
        }

        #ifdef DEBUG
        static long unsigned debug = 0;
        if (debug++ % 1000000 == 0)
            std::cout << std::bitset<8>(error) << std::endl;
        #endif

        return error;
    }

    void UpdateErrors(int depth) {
        /*
        // Read errors
        for (int i = 0; i < DATATYPES; i++) {
            for (int j = 1; j < buffers[i].size(); j++) {
                if (depth == buffers[i][j].mlStart) {
                    UpdateReadErrors(buffers[i][j-1]);
                }
            }
        }

        // Write Errors
        */

        for (auto& updateCallback : mapping[depth].updateCallbacks)
            updateCallback();
    }

    private:


    Dim IdxToDim(std::string idx) {
        const std::string lookup = "ncmpqrs";
        return static_cast<Dim>(lookup.find(idx));
    }

    // Da aggiornare al cambiamento del primo loop del buffer superiore
    void UpdateReadErrors(Buffer& buf) {
        std::bernoulli_distribution bernoulli(buf.readBer);

        for (size_t i = 0; i < buf.readError.size(); i++) {
            uint8_t error = 0;
            for (int j = 0; j < 8; j++)
                if (bernoulli(buf.random))
                    error ^= 1UL << j;
            buf.readError[i] = error;
        }
    }


    void UpdateWriteErrors(Buffer& buf) {
        std::bernoulli_distribution bernoulli(buf.writeBer);

        for (size_t i = 0; i < buf.writeError.size(); i++) {
            uint8_t error = 0;
            for (int j = 0; j < 8; j++)
                if (bernoulli(buf.random))
                    error ^= 1UL << j;
            buf.writeError[i] = error;
        }
    }



};

/*
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
*/

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






// Aggiunge loop dimensioni mancanti perchè unitarie.
/*
Mapping mcopy(mapping);
for (int d = 0; d < Dim::DIMENSIONS; d++) {
    bool found = false;
    for (auto& loop : mcopy)
        if (loop.dim == (Dim)d)
            found = true;
    if (!found)
        mapping.emplace_back((Dim)d, mapping.back().level, 1, false, true);
}
*/