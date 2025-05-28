import org.junit.After;
import org.junit.Before;
import org.junit.experimental.runners.Enclosed;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.*;

public class ShortestPathTest {
    private TextToGraph graph;

    @Before
    public void setUp() {
        graph = new TextToGraph();
        graph.addNode("A");
        graph.addNode("B");
        graph.addNode("C");
        graph.addNode("D");
        // 构建测试图：A -> B -> D, A -> C -> D
        graph.addEdge("A", "B");
        graph.addEdge("B", "D");
        graph.addEdge("A", "C");
        graph.addEdge("C", "D");
    }

    // 测试用例1
    @Test
    public void testNoPathExists() {
        graph.addNode("X");
        assertNull(graph.calcShortestPath("A", "X", 1, "A")); // 行354-355
    }

    // 测试用例2
    @Test
    public void testSingleShortestPath() {
        List<List<Object>> result = graph.calcShortestPath("A", "B", 2, "A");
        assertEquals(1, result.size()); // 验证单路径
        assertTrue(new File("./graph/directed_graph_shortest2.png").exists()); // 行373-376
    }

    // 测试用例3
    @Test
    public void testMultipleShortestPaths() {
        List<List<Object>> result = graph.calcShortestPath("A", "D", 3, "A");
        assertEquals(2, result.size()); // A->B->D 和 A->C->D
        // 验证路径长度相同
        int len1 = (int) result.get(0).get(1);
        int len2 = (int) result.get(1).get(1);
        assertEquals(len1, len2); // 行368-371
    }
    @Test
    public void testSameNodePath() {
        // 1. 准备测试数据
        //graph.addNode("A");  // 确保节点存在

        // 2. 执行测试（A到A的最短路径）
        List<List<Object>> result = graph.calcShortestPath("A", "A", 4, "A");

        // 3. 验证结果
        if (result == null) {
            System.out.println("A到A没有路径（预期行为）");
        } else {
            System.out.println("A到A的最短路径数: " + result.size());
            for (List<Object> path : result) {
                System.out.println("路径: " + path.get(0) + " 长度: " + path.get(1));
            }
        }



        //情况2：如果允许零长度自环路径
        assertEquals(1, result.size());
        assertEquals(0, result.get(0).get(1));
    }
    @After
    public void cleanUp() {
        // 删除测试生成的临时文件
        new File("./graph/directed_graph_shortest2.dot").delete();
        new File("./graph/directed_graph_shortest3.png").delete();
    }
}