#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

int main() {
    std::string text = "hello world hello cpp world";
    std::map<std::string, int> wordCount;
    std::set<std::string> uniqueWords;

    std::istringstream stream(text);
    std::string word;

    // Count word occurrences
    while (stream >> word) {
        wordCount[word]++;
        uniqueWords.insert(word);
    }

    // Display word count
    std::cout << "Word Counts:" << std::endl;
    for (const auto& pair : wordCount) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Display unique words
    std::cout << "Unique Words: ";
    for (const auto& w : uniqueWords) {
        std::cout << w << " ";
    }
    std::cout << std::endl;

    return 0;
}
