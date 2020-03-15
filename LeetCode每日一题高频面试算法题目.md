## LeetCode每日一题高频面试算法题目

### day01 （队列实现栈）

<img src="images/day01_1.png" style="zoom:80%;" />

```java
class MyStack {
    private Queue<Integer> q1; //输入
    private Queue<Integer> q2; //输出
    private int top;
    /** Initialize your data structure here. */
    public MyStack() {
        q1 = new LinkedList<Integer>();
        q2 = new LinkedList<Integer>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        q1.offer(x); //q1接收数据，将q2中所有数据复制到q1，保证先来的数据在后面，新来的数据在前面
        while(!q2.isEmpty()){
            q1.offer(q2.poll()); //poll返回链表头并删除   offer插入链表尾部
        }
        Queue temp = q1;
        q1 = q2;
        q2 = temp;
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        return q2.poll();
    }
    
    /** Get the top element. */
    public int top() {
        return q2.peek();
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return q2.isEmpty();
    }
}

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */
```



### day02（反转一个单链表）

<img src="images/day02_1.png" style="zoom: 80%;" />

```java
非递归
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next==null)
            return head;
        ListNode curNode = head.next;
        ListNode preNode = head;
        preNode.next = null;
        ListNode curNext;
        while(curNode!=null){
            curNext = curNode.next;
            curNode.next = preNode;
            preNode = curNode;
            curNode = curNext;
        }
        return preNode;
    }
}
时间复杂度：O(n)，假设 nn 是列表的长度，时间复杂度是 O(n)。
空间复杂度：O(1)。
```

<img src="images/day02_2.png" style="zoom:80%;" />

```java
//1->2->3->4->5:递归执行完向下走的时候，第一次的p指向5，head指向4,head.next是5，当执行head.next.next=head时，p.next指向4，当执行head.next=null时，断开head的4到5的节点完成一次反转，以此类推
public ListNode reverseList(ListNode head){
    if(head==null || head.next==null){
        return head;
    }
    ListNode p = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return p;
}
时间复杂度：O(n)，假设 nn 是列表的长度，那么时间复杂度为 O(n)。
空间复杂度：O(n)，由于使用递归，将会使用隐式栈空间。递归深度可能会达到 n层。
```

### day03（合并两个排序数组保证有序）

<img src="images/day03_1.png" style="zoom:80%;" />

解析：

<img src="images/day03_2.png" style="zoom: 67%;" />

```java
class Solution {
    public void merge(int[] A, int m, int[] B, int n) {
        int pa = 0;
        int pb = 0;
        int[] sorted = new int[m+n];
        int curVal = 0;
        while(pa<m || pb<n){
            if(pa==m){            //注意判断其中一个是否到达尾部
                curVal = B[pb++];
            }else if(pb==n){
                curVal = A[pa++];
            }else if(A[pa]<B[pb]){
                curVal = A[pa++];
            }else{
                curVal = B[pb++];
            }
           sorted[pa+pb-1] = curVal;
        }
        for(int i=0;i<m+n;i++){
            A[i] = sorted[i];
        }
    }
}
```

### day04（腐烂的橘子）

<img src="images/day04_1.png" style="zoom: 80%;" />

1、先将所有腐烂橘子放入Queue（LinkedList）中，建立Map  key=r*C+c  value=此时此刻腐烂橘子所经历的时间

2、当queue不为空 循环遍历，queue  remove得到腐烂橘子队列中的位置，分析该腐烂橘子上下左右使其腐烂，并把腐烂橘子（key=r*C+c, value=上层腐烂橘子对应时间+1）

3、遍历网格，如果有位置为1，说明有橘子未腐烂，return -1，否则返回map中的最大value

```java
class Solution {
    //对行和列进行移动，上，左，下，右
    int[] dr = new int[]{-1,0,1,0};
    int[] dc = new int[]{0,-1,0,1};
    public int orangesRotting(int[][] grid) {
        int R = grid.length;
        int C = grid[0].length;
        
        Queue<Integer> queue = new LinkedList();
        Map<Integer,Integer> depth = new HashMap<>();
        //先遍历寻找该开始就腐烂的橘子
        for(int r=0;r<R;r++){
            for(int c=0;c<C;c++){
                if(grid[r][c]==2){
                    int code = r*C+c; //将表格中腐烂橘子的二维坐标转化为一个数字编码
                    queue.add(code);
                    depth.put(code,0); //key为二维坐标对应的数字编码，value为该编码对应的橘子腐烂用时
                }
            }
        }
        
        int ans = 0;
        while(!queue.isEmpty()){
            int code = queue.remove();
            int r = code/C;
            int c = code%C;
            for(int k=0;k<4;k++){  //将该腐烂橘子的上下左右依次腐烂
                int nr = r + dr[k];
                int nc = c + dc[k];
                if(nr>=0 && nr<R && nc>=0 && nc<C && grid[nr][nc]==1){
                    grid[nr][nc] = 2;
                    int ncode = nr*C+nc;
                    queue.add(ncode);
                    depth.put(ncode,depth.get(code)+1); //对腐烂橘子的时刻进行重新设定  注意depth.get(code)不是ncode 
                    ans = depth.get(ncode);
                }
            }
        }
        
        for(int[] r:grid){
            for(int c :r){
                if(c==1){
                    return -1;
                }
            }
        }
        return ans;
        
    }
}

时间复杂度：O(nm)
即进行一次广度优先搜索的时间，其中 n=grid.lengthn=grid.length, m=grid[0].lengthm=grid[0].length 

空间复杂度：O(nm)
需要额外的 disdis 数组记录每个新鲜橘子被腐烂的最短时间，大小为 O(nm)，且广度优先搜索中队列里存放的状态最多不会超过 nmnm 个，最多需要 O(nm) 的空间，所以最后的空间复杂度为 O(nm)。

```

### day05（分糖果II）

<img src="images/day05_1.png" style="zoom:80%;" />

```java
class Solution {
    public int[] distributeCandies(int candies, int num_people) {
        int[] ans = new int[num_people];
        int i = 0;
        while(candies!=0){
            ans[i%num_people] += Math.min(candies,i+1);  //这里i%num_people用来多次循环数组
            candies -= Math.min(candies,i+1);
            i +=1;
        }
        return ans;
    }
}
```

### day06（和为s的连续正数序列）

<img src="images/day06_1.png" style="zoom:80%;" />

Idea:（滑动数组）

1、因为target>=1 所以滑动数组左右边界初始化为1

2、规定滑动数组尺寸[i,j)  左开右闭

3、循环次数为 i<=target/2  因为数组连续，当滑动数组最左边>=target/2时，数组无论如何也不会出现和为target组合

```java
class Solution {
    public int[][] findContinuousSequence(int target) {
        //滑动窗口解决问题
        int i = 1;  //窗口最左端
        int j = 1;  //窗口最右端  [i,j)
        List<int[]> ans = new ArrayList<>();
        int sum = 0;
        while(i<=target/2){
            if(sum<target){
                sum += j;
                j++;
            }else if(sum>target){
                sum -= i;
                i++;
            }else{
                int[] temp = new int[j-i];
                for(int k=i;k<j;k++){
                    temp[k-i] = k; 
                }
                ans.add(temp);
                sum -= i;
                i++;
            }
        }
        return ans.toArray(new int[ans.size()][]);
    }
}
```

### day07（队列的最大值）

<img src="images/day07_1.png" style="zoom:80%;" />

idea:

1、初始化一个队列存放每次push_back的数据

2、初始化一个双端队列，要求该队列front-->back为递减，保证front始终为最大值

```java
class MaxQueue {
    private Queue<Integer> queue = new LinkedList<>();
    private Deque<Integer> deque = new ArrayDeque<>();
    public MaxQueue() {
       
    }
    
    public int max_value() {
       return deque.isEmpty() ? -1:deque.peekFirst();
    }
    
    public void push_back(int value) {
        queue.offer(value);
        //deque成为递减队列
        while(!deque.isEmpty()&&deque.peekLast()<value){
            deque.pollLast();
        }
        deque.offerLast(value);
    }
    
    public int pop_front() {
        Integer result = queue.poll();  //queue存放入队列的各个数据
        if(!deque.isEmpty()&&deque.peekFirst().equals(result)){
            deque.pollFirst();
        }
        return result==null ? -1:result;
    }
}

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue obj = new MaxQueue();
 * int param_1 = obj.max_value();
 * obj.push_back(value);
 * int param_3 = obj.pop_front();
 */
```

### day08（零钱兑换）

<img src="images/day08_1.png" style="zoom:80%;" />

Idea:

<img src="images/day08_2.png" style="zoom:80%;" />

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int max = amount+1;
        int[] dp = new int[amount+1];
        
        Arrays.fill(dp,max);
        dp[0] = 0;
        for(int i=1;i<=amount;i++){
            for(int j=0;j<coins.length;j++){
                if(coins[j]<=i){
                    dp[i] = Math.min(dp[i],dp[i-coins[j]]+1);
                }
            }
        }
        
        return dp[amount]>amount ? -1:dp[amount];
    }
}
```

### day09（买卖股票系列）

<img src="images/day09_1.png" style="zoom:80%;" />

```java
class Solution {
    public int maxProfit(int[] prices) {
        //用minPriceBefore记录截止当前天的最低历史价格
        int minPriceBefore = Integer.MAX_VALUE;
        int maxProfit = 0;
        for(int i=0;i<prices.length;i++){
            if(prices[i]<minPriceBefore){
                minPriceBefore = prices[i];
            }else if(prices[i]-minPriceBefore>maxProfit){
                maxProfit = prices[i]-minPriceBefore;
            }
        }
        return maxProfit;
    }
}
```

<img src="images/day09_2.png" style="zoom:80%;" />

我们可以直接继续增加加数组的连续数字之间的差值，如果第二个数字大于第一个数字，我们获得的总和将是最大利润。这种方法将简化解决方案。

[1,7,2,3,6,7,6,7]

<img src="images/day09_3.png" style="zoom:80%;" />

```java
class Solution {
    public int maxProfit(int[] prices) {
        int maxprofit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1])
                maxprofit += prices[i] - prices[i - 1];
        }
        return maxprofit;
    }
}
```

### day10（树的遍历系列）

<img src="images/day10_1.png" style="zoom:80%;" />

```java
递归：
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};
*/
//前序遍历
class Solution {
    List<Integer> NodeList = new LinkedList<>();
    public List<Integer> preorder(Node root) {
        if(root==null){
            return NodeList;
        }

        NodeList.add(root.val);
        for(Node child:root.children){
            preorder(child);
        }
        return NodeList;
    }
}

//后序遍历
class Solution {
    List<Integer> ans = new LinkedList<Integer>();
    public List<Integer> postorder(Node root) {
        if(root == null){
            return ans;
        }
        for(Node child:root.children){
            postorder(child);
        }
        ans.add(root.val);
        return ans;
    }
}

//非递归--前序遍历
class Solution {
    public List<Integer> preorder(Node root) {
 		List<Integer> stack = new LinkedList<>();
        List<Integer> output = new LinkedList<>();
        if(root == null){
            return output;
        }
        
        stack.add(root.val);
        while(!stack.isEmpty()){
            Node node = stack.pollLast();
            output.add(node.val);
            Collections.reverse(noed.children);
           	for(Node temp:node.children){
                stack.add(temp);
            }
        }
        return output;
    }
}
```

<img src="images/day10_2.png" style="zoom:80%;" />

<img src="images/day10_3.png" style="zoom:80%;" />

```java
class Solution {
    int ans;
    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }
    public int depth(TreeNode node) {
        if (node == null) return 0; // 访问到空节点了，返回0
        int L = depth(node.left); // 左儿子为根的子树的深度
        int R = depth(node.right); // 右儿子为根的子树的深度
        ans = Math.max(ans, L+R+1); // 计算d_node即L+R+1 并更新ans
        return Math.max(L, R) + 1; // 返回该节点为根的子树的深度
    }
}
```

### day11（将数组分成和相等的三部分）

<img src="images/day11_1.png" style="zoom:80%;" />

```java
class Solution {
    public boolean canThreePartsEqualSum(int[] A) {
        int sum = 0;
        for(int i:A){
            sum+=i;
        }
        if(sum%3!=0){
            return false;
        }
        
        int s = 0;
        int flag = 0;
        for(int i=0;i<A.length;i++){
            s+=A[i];
            if(sum/3==s){
                flag++;
                s = 0;
            }
        }
        return flag>=3;  //防止[10,-10,10,-10,10,-10,10,-10]sum=0 可能导致flag>3
        
    }
}
```

### day12（字符串的最大公因子）

<img src="images/day12_1.png" style="zoom:80%;" />

```java
class Solution {
    
    public int gcd(int len1,int len2){
        int big = len1>len2 ? len1:len2;
        int small = len1<len2 ? len1:len2;
        if(big%small==0){
            return small;
        }
        return gcd(big%small,small);
    }
    public String gcdOfStrings(String str1, String str2) {
        if(!(str1+str2).equals(str2+str1)){
            return "";
        }
        int gcdStr = gcd(str1.length(),str2.length());
        return str1.substring(0,gcdStr);
    }
}
```

### day13（多数元素）

<img src="images/day13_1.png" style="zoom:80%;" />

```java
class Solution {
    public int majorityElement(int[] nums) {
        Map<Integer,Integer> record = new HashMap<>();
        int n  = nums.length;
        for(int i=0;i<n;i++){
            if(!record.containsKey(nums[i])){
                record.put(nums[i],1);
            }else{
                record.put(nums[i],record.get(nums[i])+1);
            }
        }
        int maxValue = 0;
        int ans = 0;
        for(Map.Entry<Integer,Integer> entry:record.entrySet()){
            if(entry.getValue()>maxValue){
                maxValue = entry.getValue();
                ans = entry.getKey();
            }
        }
        return ans;
}
}
```

遍历Map的四种方式

```java
public static void main(String[] args) {
        // 循环遍历Map的4中方法
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        map.put(1, 2);
        // 1. entrySet遍历，在键和值都需要时使用（最常用）
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            System.out.println("key = " + entry.getKey() + ", value = " + entry.getValue());
        }
        // 2. 通过keySet或values来实现遍历,性能略低于第一种方式
        // 遍历map中的键
        for (Integer key : map.keySet()) {
            System.out.println("key = " + key);
        }
        // 遍历map中的值
        for (Integer value : map.values()) {
            System.out.println("key = " + value);
        }
        // 3. 使用Iterator遍历
        Iterator<Map.Entry<Integer, Integer>> it = map.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Integer, Integer> entry = it.next();
            System.out.println("key = " + entry.getKey() + ", value = " + entry.getValue());
        }

        // 4. java8 Lambda
        // java8提供了Lambda表达式支持，语法看起来更简洁，可以同时拿到key和value，
        // 不过，经测试，性能低于entrySet,所以更推荐用entrySet的方式
        map.forEach((key, value) -> {
            System.out.println(key + ":" + value);
        });
        
    }
```

### day14（最长上升子序列）

<img src="images/day14_2.png" alt="动态规划问题" style="zoom:80%;" />

思路：

<img src="images/day14_3.png" style="zoom:50%;" />

<img src="images/day14_4.png" style="zoom:50%;" />

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        //动态规划
        if(nums.length==0){
            return 0;
        }
        //dp[i]代表数组中第<=i的子数组中最长上升子序列长度
        int[] dp = new int[nums.length];
        int maxAns = 1;
        for(int i=0;i<nums.length;i++){
            int maxBeforeI = 0; //定义dp[<i]中最大的值，即子数组最长上升子序列长度
            dp[i] = 1;
            for(int j=0;j<i;j++){
                //找到nums[i]>nums[j]时候  dp[1..j]中的最大值
                //只有nums[i]>nums[i]的时候，才会出现最长生长子序列递增，才去更新dp
                if(nums[i]>nums[j]){
                    maxBeforeI = Math.max(maxBeforeI,dp[j]);
                }
            }
            dp[i] = maxBeforeI+1;
            maxAns = Math.max(maxAns,dp[i]);
        }
        return maxAns;
    }
}
```

类似题目：

<img src="images/day14_1.png" style="zoom:80%;" />

```java
class Solution {
    public boolean increasingTriplet(int[] nums) {
        if(nums.length<3){
            return false;
        }
        int[] dp = new int[nums.length];
        for(int i=0;i<nums.length;i++){
            int maxBeforeI = 0;
            for(int j=0;j<i;j++){
                if(nums[i]>nums[j]){
                    maxBeforeI = Math.max(maxBeforeI,dp[j]);
                }
            }
            dp[i] = maxBeforeI + 1;
        }
        for(int i:dp){
            if(i>=3){
                return true;
            }
        }
        return false;
    }
}
```

### day15（岛屿的最大面积DFS）

<img src="images/day15_1.png" style="zoom:80%;" />

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int maxArea = 0;
        for(int i=0;i<grid.length;i++){
            for(int j=0;j<grid[i].length;j++){
                if(grid[i][j]==1){
                    maxArea = Math.max(maxArea,dfs(i,j,grid));   
                }
            }
        }
        return maxArea;
    }
    //递归遍历寻找最大连续1的个数（岛屿面积）
    public int dfs(int i,int j,int[][] grid){
        if(i<0 || j<0 || i>=grid.length || j>=grid[i].length || grid[i][j]==0){
            return 0;
        }
        grid[i][j]=0;//把当前[i][j]置为0  防止回溯遍历出现栈溢出
        int num = 1;
        num += dfs(i-1,j,grid);
        num += dfs(i+1,j,grid);
        num += dfs(i,j-1,grid);
        num += dfs(i,j+1,grid);
        return num;
    }
}
```

```java
//采用栈的方式   
class Solution {

    public int maxAreaOfIsland(int[][] grid) {
        Deque<int[]> stack = new LinkedList<>();

        int[][] moveIndexArray = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int maxArea = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                stack.add(new int[]{i, j});
                //计算最大面积
                int currMaxArea = 0;
                while (!stack.isEmpty()) {
                    int[] pop = stack.pop();
                    int currI = pop[0];
                    int currJ = pop[1];
                    if (currI < 0 || currI >= grid.length || currJ < 0 || currJ >= grid[0].length || grid[currI][currJ] == 0) {
                        continue;
                    }
                    currMaxArea++;
                    grid[currI][currJ] = 0;
                    for (int[] moveIndex : moveIndexArray) {
                        stack.add(new int[]{currI + moveIndex[0], currJ + moveIndex[1]});
                    }
                }
                maxArea = Math.max(currMaxArea, maxArea);
            }
        }

        return maxArea;
    }

}
```

