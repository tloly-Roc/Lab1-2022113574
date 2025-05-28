import org.junit.experimental.runners.Enclosed;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Set;

import static org.junit.Assert.*;

@RunWith(Enclosed.class)
class test3 {

    @BeforeEach
    void setUp() {
    }

    @AfterEach
    void tearDown() {
    }

    public static class BridgeWordsTest3 {
        @Test
        public void testWordNotInGraph() {
            TextToGraph graph = new TextToGraph();
            graph.addNode("hello");
            graph.addEdge("hello", "world");

            Set<String> result = graph.queryBridgeWords("hello", "nonexistent", false);
            assertTrue(result == null || result.isEmpty());
        }
    }
}