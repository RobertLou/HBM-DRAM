#include "embeddingRW.h"

int main() {
	std::string embeddingloc;
	embeddingloc = "../embedding_map/embedding.csv";
	std::vector<parameters> lines;
	
	readembedding(embeddingloc,lines,1);

	std::string newembeddingloc;
	newembeddingloc = "../embedding_map/newembedding.csv";

	writeembedding(newembeddingloc,lines,1);
}
