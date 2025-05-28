import org.junit.experimental.runners.Enclosed;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Set;

import static org.junit.Assert.*;

@RunWith(Enclosed.class)
class test4 {


    public static class BridgeWordsTest4 {
        @Test
        public void testMultipleBridgeWords() {
            TextToGraph graph = new TextToGraph();
            graph.addNode("hello");
            graph.addNode("wordA");
            graph.addNode("wordB");
            graph.addNode("world");
            graph.addEdge("hello", "wordA");
            graph.addEdge("wordA", "world");
            graph.addEdge("hello", "wordB");
            graph.addEdge("wordB", "world");

            Set<String> result = graph.queryBridgeWords("hello", "world", false);
            assertTrue(result.contains("wordA"));
            assertTrue(result.contains("wordB"));
        }
    }
}