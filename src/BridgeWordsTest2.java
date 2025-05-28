import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.*;

public class BridgeWordsTest2 {
    @Test
    public void testNoBridgeWord() {
        TextToGraph graph = new TextToGraph();
        graph.addNode("hello");
        graph.addNode("world");
        graph.addEdge("hello", "world"); // 直接连接，无桥接词

        Set<String> result = graph.queryBridgeWords("hello", "world", false);
        assertTrue(result.isEmpty());
    }
}