import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.*;

public class BridgeWordsTest {
    @Test
    public void testBridgeWordExists() {
        TextToGraph graph = new TextToGraph();
        graph.addNode("hello");
        graph.addNode("bridge");
        graph.addNode("world");
        graph.addEdge("hello", "bridge");
        graph.addEdge("bridge", "world");

        Set<String> result = graph.queryBridgeWords("hello", "world", false);
        assertTrue(result.contains("bridge"));
    }
}