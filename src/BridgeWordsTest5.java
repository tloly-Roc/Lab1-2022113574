import org.junit.experimental.runners.Enclosed;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Set;

import static org.junit.Assert.*;

@RunWith(Enclosed.class)
class test5 {



    public static class BridgeWordsTest5 {
        @Test
        public void testEmptyInput() {
            TextToGraph graph = new TextToGraph();
            graph.addNode("hello");
            graph.addEdge("hello", "world");

            Set<String> result = graph.queryBridgeWords("", "world", false);
            assertTrue(result == null || result.isEmpty());
        }
    }
}