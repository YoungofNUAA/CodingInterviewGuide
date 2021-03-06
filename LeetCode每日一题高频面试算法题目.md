## LeetCode每日一题高频面试算法题目

### day01 （队列实现栈）

<img src="images/day01_1.png" style="zoom:80%;" />

```java
 public static class QueueToStack{
        private Queue<Integer> queue;
        private Queue<Integer> help;
        public QueueToStack(){
            queue = new LinkedList<Integer>();
            help = new LinkedList<Integer>();
        }
  
        public void push(int pushInt){
            queue.add(pushInt);
        }
        /**
         * @Description: 弹栈操作操作
         * 弹栈时，queue队列所有数据迁移至 help 返回最后一个数 并交换指针
         */
        public Integer pop(){
            if (queue.isEmpty())
                throw new RuntimeException("栈空！");
            while (queue.size()>1){
                help.add(queue.poll());
            }
            int temp = queue.poll();
            swap();
            return temp;
        }
  
        /**
         * @Description: 栈的peek操作 只返回栈顶元素
         * 原理同pop操作，但是在最后的一个元素要继续入队 help 因为peek只是返回栈顶 并非弹出
         */
        public Integer peek(){
            if (queue.isEmpty())
                throw new RuntimeException("栈空");
            while (queue.size()>1){
                help.add(queue.poll());
            }
            int temp=queue.poll();
            help.add(temp); //关键地方
            swap();
            return temp;
        }
  
        private void swap() {
            Queue<Integer> temp = queue;
            queue = help;
            help = temp;
        }
    }
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

2、规定滑动数组尺寸[i,j)  左闭右开

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

#### 遍历Map的四种方式

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

### day14（最长上升子序列--动态规划）

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

#### 动态规划类似题目：

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

<img src="images/day14_5.png" style="zoom:80%;" />

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for(int i=1;i<nums.length;i++){
            if(dp[i-1]>0){
                dp[i] = dp[i-1]+nums[i];
            }else{
                dp[i] = nums[i];
            }
            maxSum = Math.max(maxSum,dp[i]);
        }
        return maxSum;
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

### day16（字符串压缩--双指针）

<img src="images/day16_1.png" style="zoom:80%;" />

```java
class Solution {
    public String compressString(String S) {
        int N = S.length();
        int i=0;
        StringBuilder sb = new StringBuilder();
        while(i<N){
            int j=i;
            while(j<N && S.charAt(j)==S.charAt(i)){
                j++;
            }
            sb.append(S.charAt(i)).append(j-i);
            i = j;
        }
        
        String ans = sb.toString();
        return ans.length()>=N ? S:ans;
    }
}
```

### day17（拼写单词）

<img src="images/day17_1.png" style="zoom:80%;" />

```java
class Solution {
    public int countCharacters(String[] words, String chars) {
        int[] hash = new int[26];
        for(char c:chars.toCharArray()){
            hash[c-'a'] += 1;
        }
        
        int[] map = new int[26];
        int len = 0;
        for(int i=0;i<words.length;i++){
            String word = words[i];
            Arrays.fill(map,0);
            boolean flag = true;
            for(char c:words[i].toCharArray()){
                map[c-'a'] ++;
                if(map[c-'a']>hash[c-'a']){
                    flag = false;
                }
            }
            if(flag){
                len+=word.length();
            }else{
                len+=0;
            }
        }
        return len;
    }
}
```



### day18（矩形重叠）

<img src="images/day18_1.png" style="zoom:80%;" />

```java
//想象一下，如果我们在平面中放置一个固定的矩形 rec2，那么矩形 rec1 必须要出现在 rec2 的「四周」，也就是说，矩形 rec1 需要满足以下四种情况中的至少一种：

//矩形 rec1 在矩形 rec2 的左侧；

//矩形 rec1 在矩形 rec2 的右侧；

//矩形 rec1 在矩形 rec2 的上方；

//矩形 rec1 在矩形 rec2 的下方。
class Solution {
    public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
        //先判断不重叠的情况  再取反
        return (!(rec1[1]>=rec2[3]||rec1[3]<=rec2[1]||rec1[2]<=rec2[0]||rec1[0]>=rec2[2]));
    }
}
```

### day19（最长回文串）

<img src="images/day19_1.png" style="zoom:80%;" />

对于每个字符 `ch`，假设它出现了 `v` 次，我们可以使用该字符 `v / 2 * 2` 次，在回文串的左侧和右侧分别放置 `v / 2` 个字符 `ch`，其中 `/` 为整数除法。例如若 `"a"` 出现了 `5` 次，那么我们可以使用 `"a"` 的次数为 `4`，回文串的左右两侧分别放置 `2` 个 `"a"`。

如果有任何一个字符 `ch` 的出现次数 `v` 为奇数（即 `v % 2 == 1`），那么可以将这个字符作为回文中心，注意只能最多有一个字符作为回文中心。在代码中，我们用 `ans` 存储回文串的长度，由于在遍历字符时，`ans` 每次会增加 `v / 2 * 2`，因此 `ans` 一直为偶数。但在发现了第一个出现次数为奇数的字符后，我们将 `ans` 增加 `1`，这样 `ans` 变为奇数，在后面发现其它出现奇数次的字符时，我们就不改变 `ans` 的值了。

```

```

```java
class Solution {
    public int longestPalindrome(String s) {
        int[] arr = new int[128];
        for(char c:s.toCharArray()){
            arr[c]++;
        }
        
        int ans = 0;
        for(int n:arr){
            ans+=n/2*2;
            //第一次遇到奇数字符时，ans+1,以后遇到奇数不再增加ans直接跳过，只有偶数增加在ans左右两端         
           //ans%2==0  防止以后遍历再次进入if语句
            if(n%2==1 && ans%2==0 ){
                ans ++;
            }
        }
        return ans;
    }
}
```

### day20（最小的k个数）

<img src="images/day20_1.png" style="zoom:80%;" />

相等于练习冒泡排序吧 哈哈哈

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        sortArray(arr);
        int[] ans = new int[k];
        for(int i=0;i<k;i++){
            ans[i] = arr[i];
        }
        return ans;
    }
    
    public void sortArray(int[] arr){
        for(int i=0;i<arr.length;i++){
            boolean flag = true;
            for(int j=0;j<arr.length-i-1;j++){
                if(arr[j]>arr[j+1]){
                    flag = false;
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
            if(flag){
                break;
            }
        }
    }
}
```

### day21（水壶问题--最大公约数）

<img src="images/day21_1.png" style="zoom:80%;" />

idea:

```
若a,b是整数,且gcd(a,b)=d，那么对于任意的整数x,y,ax+by都一定是d的倍数，特别地，一定存在整数x,y，使ax+by=d成立。
     * 裴蜀定理：
     * 如果所需要的水量是两个水壶容量的最大公约数的倍数，且水量不大于两个水壶的容量之和，那么必然可以用这两个水壶操作得到所需要的水量。
```

```java
class Solution {
    public boolean canMeasureWater(int x, int y, int z) {
        //解决x或者y中一个为0的情况
        if(x==0||y==0){
            if(z==x||z==y){
                return true;
            }
            return false;
        }
        if(x+y<z){
            return false;
        }
        int temp = gcd(x,y);
        return z%temp==0;
    }
    
    private int gcd(int x,int y){
        int big = x>y? x:y;
        int small = x<y? x:y;
        if(big%small==0){
            return small;
        }
        return gcd(big%small,small);
    }
}
```

### day22（使数组唯一的最小增量）

<img src="images/day22_1.png" style="zoom:80%;" />

```java
class Solution {
    public int minIncrementForUnique(int[] A) {
        //先排序
        Arrays.sort(A);
        int ans = 0;
        for(int i=1;i<A.length;i++){
            if(A[i]<=A[i-1]){
                int temp = A[i];
                A[i] = A[i-1]+1;
                ans+=A[i-1]+1-temp;
            }
        }
        return ans;
    }
}
```

### day23（链表中间节点--快慢指针）

<img src="images/day23_1.png" style="zoom:80%;" />

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode middleNode(ListNode head) {
        if(head==null){
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while(fast!=null && fast.next!=null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

### day24（按摩师问题--动态规划）

<img src="images/day24_1.png" style="zoom:80%;" />

**idea:**

```
dp[i][0] 第i个预约不接受的最佳总时长，则i-1可以接收也可以不接受
dp[i][1] 第i个预约接收的最佳总时长，则i-1一定为不接受
```

```java
class Solution {
    public int massage(int[] nums) {
        int n = nums.length;
        if(n==0) return 0;
        if(n==1) return nums[0];
        
        int[][] dp = new int[n][2];
        dp[0][1] = nums[0];
        dp[0][0] = 0;
        for(int i=1;i<n;i++){
            dp[i][1] = dp[i-1][0] + nums[i];
            dp[i][0] = Math.max(dp[i-1][1],dp[i-1][0]);
        }
        return Math.max(dp[n-1][0],dp[n-1][1]);
    }
}
```

### day25（三维形体表面积）

<img src="images/day25_1.png" style="zoom:80%;" />

Idea:

1、输入二维数组，数字代表数组i,j位置放置的立方体个数

2、要点是要把i,j方向上立方体重叠面积进行删除

```java
class Solution {
    public int surfaceArea(int[][] grid) {
        int n = grid.length;
        int area = 0;
        for(int i = 0;i<n;i++){
            for(int j=0;j<grid[i].length;j++){
                int level = grid[i][j];
                if(level>0){
                    area += (level<<2)+2;
                }
                //删除i,j方向上的重叠面积
                if(i>0) area -= Math.min(grid[i-1][j],grid[i][j])*2;
                if(j>0) area -= Math.min(grid[i][j-1],grid[i][j])*2;
            }
        }
        return area;
    }
}
```

### day26（车的可用捕获量）

<img src="images/day26_1.png" style="zoom:80%;" />

<img src="images/day26_2.png" style="zoom:80%;" />

idea：

题目太长迷惑人心，要点如下：

1、先找到白色车的位置 i, j

2、在四个方向上移动车，车停止条件为 a: 超出8x8棋盘界限  b: 遇到白色的象 c: 遇到要找的黑色卒

```java
class Solution {
    public int numRookCaptures(char[][] board) {
        //上下左右
        int[] dx = {-1,1,0,0};
        int[] dy = {0,0,-1,1};
        
        int ans = 0;
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                //找到白色车的位置
                if(board[i][j]=='R'){
                    //四个方向进行前进
                    for(int k=0;k<4;k++){
                        //把白色车的位置赋值给移动变量
                        int moveX = i;
                        int moveY = j;
                        while(true){
                            moveX += dx[k];
                            moveY += dy[k];
                            //判断是否越界  车不能与其他友方（白色）象进入同一个方  
                            if(moveX<0 || moveX>=8 || moveY<0 || moveY>=8 || board[moveX][moveY]=='B'){
                                break;
                            }
                            
                            if(board[moveX][moveY]=='p'){
                                ans++;
                                break; //在另一个方向查找
                            }
                        }
                    }
                }
            }
        }
        return ans;
    }
}
```

### day27（卡牌分组）

<img src="images/day27_1.png" style="zoom:80%;" />

idea:

求数组中各个相同元素个数的最大公约数，如果最大公约数>=2则返回true  否则返回false

```java
class Solution {
    public boolean hasGroupsSizeX(int[] deck) {
        int[] counter = new int[10000];
        for(int i=0;i<deck.length;i++){
            counter[deck[i]] ++;
        }
        int x =0;
        for(int count:counter){
            if(count>0){
                //计算最大公约数
                x = gcd(x,count);
                if(x==1){
                    return false;
                }
            }
        }
        return x>=2;
    }
    
    private int gcd(int a,int b){
        if(a==0) return b;
        if(b==0) return a;
        int max = a>b? a:b;
        int min = a<b? a:b;
        if(max%min==0){
            return min;
        }
        return gcd(max%min,min);
    }
    
}
```

### BFS/DFS问题

#### 快速幂算法

```java
 
    public static long pow(int n) {
        if (n == 0)
            return 1;
        long half = pow(n / 2);
        if (n % 2 == 0)
            return (half * half);
        else
            return (half * half * 2);
    }
```

#### BFS地图问题（阿里巴巴面试题）

 一个地图n*m，包含1个起点，1个终点，其他点包括可达点和不可达点。 每一次可以：上下左右移动，或使用1点能量从（i,j)瞬间移动到（n-1-i, m-1-j)，最多可以使用5点能量。

```java
package cn.nuaa.alibaba;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class test02 {
    static int[] dx = {1,-1,0,0};
    static int[] dy = {0,0,1,-1};
    static int m;
    static int n;
    static int endX;
    static int endY;
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        n = scanner.nextInt();
        m = scanner.nextInt();
        char[][] map = new char[n][m];
        Queue<Pair> queue = new LinkedList<>();
        for(int i=0;i<n;i++){
            map[i] = scanner.next().toCharArray();
            for(int j=0;j<map[i].length;j++){
                if(map[i][j]=='S'){
                    Pair pair = new Pair(i,j);
                    queue.add(pair);
                }else if(map[i][j]=='E'){
                    endX = i;
                    endY = j;
                }
            }
        }
        //BFS
        System.out.println(BFS(map,queue));
    }
    public static boolean check(int x,int y){
        if(x<0 || y<0 || x>=n || y>=m){
            return false;
        }
        return true;
    }
    public static int BFS(char[][] map,Queue<Pair>queue){
        while (!queue.isEmpty()){
            int size = queue.size();
            while (size-- >0){
                Pair top = queue.poll();
                if(top.x==endX && top.y==endY){
                    return top.step;
                }
                for(int k=0;k<4;k++){
                    int curX = top.x+dx[k];
                    int curY = top.y+dy[k];
                    Pair nextPair = new Pair(curX,curY);
                    nextPair.step = top.step + 1;
                    nextPair.fly = top.fly;
                    if(check(curX,curY) && (map[curX][curY]=='.' || map[curX][curY]=='E')){
                        queue.add(nextPair);
                        map[curX][curY] = 'X';
                    }
                }
                int flyX = n-1-top.x;
                int flyY = m-1-top.y;
                if(check(flyX,flyY) && top.fly<5 && (map[flyX][flyY]=='.' || map[flyX][flyY]=='E')){
                    Pair pair = new Pair(flyX,flyY);
                    pair.step = top.step+1;
                    pair.fly = top.fly+1;
                    queue.add(pair);
                    map[flyX][flyY] = 'X';
                }
            }
        }
        return -1;
    }
}

class Pair{
    int x;
    int y;
    int step;
    int fly;

    public Pair(int x, int y) {
        this.x = x;
        this.y = y;
    }
}

```

#### LeetCode BFS问题

##### 被围绕的区域

<img src="images/BFS.png" style="zoom:80%;" />

**总结：**

**1、一般需要定义一个内部类，代表地图每个点，将题目中给的属性加进去，基础属性为坐标x,y**

**2、BFS问题需要queue，DFS问题需要stack**

**3、一般需要定义上下左右坐标转移数组**  

**4、需要定义坐标范围检查函数**

```java
class Solution {
    private class Node{
        int x;
        int y;
        public Node(int x,int y){
            this.x = x;
            this.y = y;
        }
    }
    
    int m = 0;
    int n = 0;
    public void solve(char[][] board) {
        if(board.length==0 || board==null){
            return;
        }
        m = board.length;
        n = board[0].length;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                boolean isEdge = (i==0 || i==m-1 ||j==0 ||j==n-1) ? true:false;
                if(isEdge && board[i][j]=='O'){
                    bfs(board,i,j);
                }
            }
        }
        
        //与边界相连的O用#代替  
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(board[i][j] == 'O'){
                    board[i][j] = 'X';
                }
                if(board[i][j] == '#'){
                    board[i][j] = 'O';
                }
            }
        }
    }
    
    public boolean check(int x,int y){
        if(x<0 || x>=m || y<0 || y>=n){
            return false;
        }
        return true;
    }
    
    public void bfs(char[][] board,int i,int j){
        Queue<Node> queue = new LinkedList<>();
        queue.add(new Node(i,j));
        board[i][j] = '#';
        //上下左右
        int[] dx = {-1,1,0,0};
        int[] dy = {0,0,-1,1};
        while(!queue.isEmpty()){
            Node curNode = queue.poll();
            
            for(int k=0;k<4;k++){
                int nextX = curNode.x + dx[k];
                int nextY = curNode.y + dy[k];
                if(check(nextX,nextY) && board[nextX][nextY]=='O'){
                    queue.add(new Node(nextX,nextY));
                    board[nextX][nextY] = '#';
                }
            }
        }
    }
}
```

##### 最远海洋问题

参考day29 ☟

### 动态规划问题



### day28（单词的压缩编码---set存储不重复后缀）

<img src="images/day28_1.png" style="zoom:80%;" />

idea:

如果单词 X 是 Y 的后缀，那么单词 X 就不需要考虑了，因为编码 Y 的时候就同时将 X 编码了。例如，如果 words 中同时有 "me" 和 "time"，我们就可以在不改变答案的情况下不考虑 "me"。

如果单词 Y 不在任何别的单词 X 的后缀中出现，那么 Y 一定是编码字符串的一部分。

因此，目标就是保留所有不是其他单词后缀的单词，最后的结果就是这些单词长度加一的总和，因为每个单词编码后后面还需要跟一个 # 符号。

```java
class Solution {
    public int minimumLengthEncoding(String[] words) {
        Set<String> set = new HashSet<String>(Arrays.asList(words));
        for(String word:words){
            for(int i=1;i<word.length();i++){ //i一定从1开始，否则会把该单词从set中移除
                set.remove(word.substring(i));
            }
        }
        
        int ans = 0;
        // for(String word:set){
        //     ans += word.length()+1；
        // }
        Iterator<String> iter = set.iterator();
        while(iter.hasNext()){
            String temp = iter.next();
            ans += temp.length()+1;
        }
        return ans;
    }
}
```

### day29（BFS最远海洋问题）

<img src="images/day29_1.png" style="zoom:80%;" />

Idea:

我们只要先把所有的陆地都入队，然后从各个陆地**同时开始**一层一层的向海洋扩散，那么最后扩散到的海洋就是最远的海洋！并且这个海洋肯定是被离他最近的陆地给扩散到的！

<img src="images/day29_2.png" style="zoom:80%;" />

```java
class Solution {
    
    int m;
    int n;
    
    //定义内部类标记一些题目要求的属性
    private class Node{
        int x;
        int y;
        int far;  //定义该点距离出发点大陆的距离
        public Node(int x,int y){
            this.x = x;
            this.y = y;
        }
    }
    
    public boolean check(int x,int y){
        if(x<0 || x>=m || y<0 || y>=n){
            return false;
        }
        return true;
    }
    
    public int maxDistance(int[][] grid) {
        //首先遍历grid找出所有为1的元素
        
        m = grid.length;
        n = grid[0].length;
        
        Queue<Node> queue = new LinkedList<>();
        
        for(int i=0;i<m;i++){
            for(int j = 0;j<n;j++){
                if(grid[i][j]==1){
                    queue.add(new Node(i,j));
                }
            }
        }
        return BFS(queue,grid);
    }
    
    public int BFS(Queue<Node> queue,int[][] grid){
        int[] dx = {-1,1,0,0};
        int[]dy = {0,0,-1,1};
        boolean hasOcean = false;
        Node newNode = null;
        while(!queue.isEmpty()){
            Node curNode = queue.poll();
            for(int k=0;k<4;k++){
                int newX = curNode.x+dx[k];
                int newY = curNode.y+dy[k];
                if(check(newX,newY) && grid[newX][newY]==0){
                    newNode = new Node(newX,newY);
                    newNode.far = curNode.far+1;  //一定是curNode.step
                    grid[newX][newY] = 1;
                    hasOcean = true;
                    queue.add(newNode);
                }
            }   
        }
        if(!hasOcean){
            return -1;
        }
        return newNode.far;
    }
}
```

### day30（圆圈中最后剩下的数字）

<img src="images/day30_1.png" style="zoom:80%;" />

```java
class Solution {
    public int lastRemaining(int n, int m) {
        List<Integer> list = new ArrayList<>();
        for(int i=0;i<n;i++){
            list.add(i);
        }
        int start = 0;
        while(list.size()>1){
            int size = list.size();
            int delIndex = (start+m-1)%size;
            list.remove(delIndex);
            start = delIndex;
        }
        return list.get(0);
    }
}
```

### day31（数组升序--冒泡、选择、插入、快速、归并、堆排序算法）

<img src="images/day31_1.png" style="zoom:80%;" />

<img src="images/day31_2.png" style="zoom:80%;" />

```java
class Solution {
    public int[] sortArray(int[] nums) {
        // BubbleSort(nums);
        // selectSort(nums);
        insertSort(nums);
        return nums;
    }
    
    //冒泡排序
    public void BubbleSort(int[] nums){
        int n = nums.length;
        for(int i=0;i<n-1;i++){
            boolean isSort = true;
            for(int j=0;j<n-i-1;j++){
                if(nums[j+1]<nums[j]){
                    isSort = false;
                    int temp = nums[j+1];
                    nums[j+1] = nums[j];
                    nums[j] = temp;
                }
            }
            if(isSort){
                break;
            }
        }
    }
    
    //选择排序
    public void selectSort(int[] nums){
        int n = nums.length;
        //遍历n-1次 前n-1选择好了之后 最后一个元素必然满足要求
        for(int i=0;i<n-1;i++){
            int minIndex = i;
            for(int j=i+1;j<n;j++){
                if(nums[j]<nums[minIndex]){
                    minIndex = j;
                }
            }
            //交换数据把最小的元素给i
            int temp = nums[minIndex];
            nums[minIndex] = nums[i];
            nums[i] = temp;
        }
    }
    
    //插入排序
    public void insertSort(int[] nums){
        int n = nums.length;
        //循环数组 将nums[i]插入到 nums[0,i)有序区间中
        for(int i=1;i<n;i++){
            int temp = nums[i];
            int j = i;
            while(j>0 && (nums[j-1]>temp)){
                nums[j] = nums[j-1];
                j--;
            }
            nums[j] = temp;
        }
    }
    
    //快速排序 ---重点掌握
   public static void quickSort(int[] nums,int L,int R){
        if(nums.length == 0){
            return;
        }

        int i = L;
        int j = R;
        int key = nums[(i+j)/2];
        while (i<=j){
            while (nums[i]<key){
                i++;
            }
            while (nums[j]>key){
                j--;
            }
            if(i<=j){
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
                i++;
                j--;
            }
        }

        if(i<R){
            quickSort(nums,i,R);
        }
        if(j>L){
            quickSort(nums,L,j);
        }
    }
}
```

**堆排序**

```java
import java.util.Arrays;

/**
 * 堆排序demo
 */
public class HeapSort {
    public static void main(String []args){
        int []arr = {9,8,7,6,5,4,3,2,1};
        sort(arr);
        System.out.println(Arrays.toString(arr));
    }
    public static void sort(int []arr){
        //1.构建大顶堆
        for(int i=arr.length/2-1;i>=0;i--){
            //从第一个非叶子结点从下至上，从右至左调整结构
            adjustHeap(arr,i,arr.length);
        }
        //2.调整堆结构+交换堆顶元素与末尾元素
        for(int j=arr.length-1;j>0;j--){
            swap(arr,0,j);//将堆顶元素与末尾元素进行交换
            adjustHeap(arr,0,j);//重新对堆进行调整
        }

    }

    /**
     * 调整大顶堆（仅是调整过程，建立在大顶堆已构建的基础上）
     * @param arr
     * @param i
     * @param length
     */
    public static void adjustHeap(int []arr,int i,int length){
        int temp = arr[i];//先取出当前元素i
        for(int k=i*2+1;k<length;k=k*2+1){//从i结点的左子结点开始，也就是2i+1处开始
            if(k+1<length && arr[k]<arr[k+1]){//如果左子结点小于右子结点，k指向右子结点
                k++;
            }
            if(arr[k] >temp){//如果子节点大于父节点，将子节点值赋给父节点（不用进行交换）
                arr[i] = arr[k];
                i = k;
            }else{
                break;
            }
        }
        arr[i] = temp;//将temp值放到最终的位置
    }

    /**
     * 交换元素
     * @param arr
     * @param a
     * @param b
     */
    public static void swap(int []arr,int a ,int b){
        int temp=arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }
}
```

**归并排序**

```java
 public class MergeSort{
 2
 3
 4    public int[] sort(int[] sourceArray) throws Exception {
 5        // 对 arr 进行拷贝，不改变参数内容
 6        int[] arr = Arrays.copyOf(sourceArray, sourceArray.length);
 7
 8        if (arr.length < 2) {
 9            return arr;
10        }
11        int middle = (int) Math.floor(arr.length / 2);
12
13        int[] left = Arrays.copyOfRange(arr, 0, middle);
14        int[] right = Arrays.copyOfRange(arr, middle, arr.length);
15
16        return merge(sort(left), sort(right));
17    }
18
19    protected int[] merge(int[] left, int[] right) {
20        int[] result = new int[left.length + right.length];
21        int i = 0;
22        while (left.length > 0 && right.length > 0) {
23            if (left[0] <= right[0]) {
24                result[i++] = left[0];
25                left = Arrays.copyOfRange(left, 1, left.length);
26            } else {
27                result[i++] = right[0];
28                right = Arrays.copyOfRange(right, 1, right.length);
29            }
30        }
31
32        while (left.length > 0) {
33            result[i++] = left[0];
34            left = Arrays.copyOfRange(left, 1, left.length);
35        }
36
37        while (right.length > 0) {
38            result[i++] = right[0];
39            right = Arrays.copyOfRange(right, 1, right.length);
40        }
41
42        return result;
43    }
44
45}
```

**归并排序**--**方法****2**

```java
import java.util.Arrays;

public class Test12 {
    public static void main(String[] args) {
        int[] test = new int[]{7,5,8,1,2,9,4,3,6,10};
        int[] temp = new int[test.length];
        sort(test,0,test.length-1,temp);
        System.out.println(Arrays.toString(test));
    }

    public static void sort(int[] arr,int left,int right,int[] temp){
        if(left<right){
            int mid = left + (right-left)/2;
            sort(arr,left,mid,temp);
            sort(arr,mid+1,right,temp);
            merge(arr,left,mid,right,temp);
        }
    }

    private static void merge(int[] arr, int left, int mid, int right, int[] temp) {
        int i = left;
        int j = mid+1;
        int t = 0;
        while (i<=mid && j<=right){
            if(arr[i]<=arr[j]){
                temp[t++] = arr[i++];
            }else{
                temp[t++] = arr[j++];
            }
        }
        while (i<=mid){
            temp[t++] = arr[i++];
        }

        while (j<=right){
            temp[t++] = arr[j++];
        }

        t = 0;
        while (left<=right){
            arr[left++] = temp[t++];
        }
    }
}

```

### day32（有效括号问题）

<img src="images/day32_1.png" style="zoom:80%;" />

```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        char[] chars = s.toCharArray();
        for(char c:chars){
            if(stack.size()==0){
                stack.push(c);
            }else if(isPair(stack.peek(),c)){
                stack.pop();
            }else{
                stack.push(c);
            }
        }
        return stack.size()==0;
    }
    
    public boolean isPair(char a,char b){
        if(a=='(' && b==')' || a=='{' && b=='}' || a=='[' && b==']'){
            return true;
        }
        return false;
    }
}
```

<img src="images/day32_2.png" style="zoom:80%;" />

今天题目太难理解了，没意思

使 max(depth(A), depth(B)) 的可能取值最小”。这句话其实相当于让A字符串和B字符串的depth尽可能的接近。为什么呢？因为seq对应的栈上，每个左括号都对应一个深度，而这个左括号，要么是A的，要么是B的。所以，栈上的左括号只要按奇偶分配给A和B就可以啦！时间复杂度很明显是 O(n)O(n)的，空间复杂度也是O(n)O(n)（如果算返回的变量的话）。

```java
class Solution {
    public int[] maxDepthAfterSplit(String seq) {
        int[] ans = new int[seq.length()];
        
        for(int i=0;i<seq.length();i++){
            //奇数下标的( 分给A 偶数下标（ 分给B 
            //奇数下标的) 分给B 偶数下标) 分给A 
            // if(seq.charAt(i)=='('){
            //     if(i%2==0){
            //         ans[i] = 1;
            //     }else{
            //         ans[i] = 0;
            //     }
            // }else{
            //     if(i%2==0){
            //         ans[i] = 0;
            //     }else{
            //         ans[i] = 1;
            //     }
            // }
            ans[i] = seq.charAt(i)=='('? i&1:(i+1)&1;
        }
        return ans;
    }
}
```

### day33（生命游戏）

<img src="images/day33_1.png" style="zoom:80%;" />

idea：

不能每个点周围判断完之后立即更新周围点的状态，因为当前点的周围状态会是下一次研究的中心点对象，采取标志位方式  2代表：alive-->dead   -1: dead-->alive

```java
class Solution {
    public int m;
    public int n;
    public void gameOfLife(int[][] board) {
        int[] dx = {0, 0, 1, -1, 1, 1, -1, -1};
        int[] dy = {1, -1, 0, 0, 1, -1, 1, -1};
        
        m = board.length;
        n = board[0].length;
        
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                int cnt = 0; //统计当前细胞周围活细胞数量
                for(int k=0;k<8;k++){
                    int next_X = i+dx[k];
                    int next_Y = j+dy[k];
                    if(check(next_X,next_Y) && board[next_X][next_Y]>0){//标记为2或者-1只是一个标识位 在没有遍历完之前2，-1还代表他底层的1,0
                        cnt ++;
                    }
                }
                
                if(board[i][j]==1){
                    if(cnt<2 || cnt>3){
                        board[i][j] = 2; //活细胞-->死亡
                    }
                }else if(cnt==3){
                    board[i][j] = -1;  //死细胞-->复活
                }
                
            }
        }
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(board[i][j]==2){
                    board[i][j] = 0;
                }else if(board[i][j]==-1){
                    board[i][j] = 1;
                }
            }
        }
    }
    
    public boolean check(int x,int y){
        if(x<0 || x>=m || y<0 || y>=n){
            return false;
        }
        return true;
    }
}
```

### day34（字符串到整数）

<img src="images/day34_1.png" style="zoom:80%;" />

<img src="images/day34_2.png" style="zoom:80%;" />

```java
class Solution {
    public int myAtoi(String str) {
        char[] chars = str.toCharArray();
        int n = chars.length;
        int cur = 0;
        //去掉前导空格
        while(cur<n && chars[cur]==' '){
            cur++;
        }
        if(cur==n){
            return 0;
        }

        boolean negative = false;
        if(chars[cur]=='-'){
            negative = true;
            cur++;
        }else if(chars[cur]=='+'){
            cur++;
        }else if(!Character.isDigit(chars[cur])){
            return 0;
        }
        
        int ans = 0;
        int digit = 0;
        while(cur<n && Character.isDigit(chars[cur])){
            digit = chars[cur]-'0';
            if(ans>(Integer.MAX_VALUE-digit)/10){ //保证不溢出
                return ans = negative? Integer.MIN_VALUE:Integer.MAX_VALUE;
            }
            ans = ans*10+digit;
            cur++;
        }
        return negative?-ans:ans;
    }
}
```

### day35（接雨水）

<img src="images/day35_1.png" style="zoom:80%;" />

对于每一列来说，他能存的雨水量是他左边最高墙和右边最高墙中较低的那堵墙的高度减去自身墙的高度。所以可以用数组记录每列左右最高墙的高度，然后计算每一列可以存的雨水量

```java
class Solution {
    public int trap(int[] height) {
        //***************暴力解法******************
        int ans = 0;
        //从第二个柱体遍历到倒数第二个柱体
        for(int i=1;i<height.length-1;i++){
            int leftMax = 0;
            int rightMax = 0;
            for(int j=0;j<=i;j++){
                leftMax = Math.max(leftMax,height[j]);
            }
            for(int k=i;k<height.length;k++){
                rightMax = Math.max(rightMax,height[k]);
            }
            ans += Math.min(leftMax,rightMax)-height[i];
        }
        return ans;
        
        //*****************dp解法********************
        int n = height.length;
        int ans = 0;
        if(n==0){
            return 0;
        }
        int[][] dp = new int[n][2];
        //dp[i][0] dp[i][1] 表示第i柱子左右两边的最大高度(包括当前柱子高度)
        dp[0][0] = height[0];   //左
        dp[n-1][1] = height[n-1];  //右
        //填充柱子左边高度所有情况
        for(int i=1;i<n;i++){
            dp[i][0] = Math.max(dp[i-1][0],height[i]);
        }
        for(int j=n-2;j>=0;j--){
            dp[j][1] = Math.max(dp[j+1][1],height[j]);
        }
        for(int k=0;k<n;k++){
            ans += Math.min(dp[k][0],dp[k][1])-height[k];
        }
        return ans;
    }
}
```

### day37（编辑距离）

<img src="images/day37_1.png" style="zoom:80%;" />
<img src="images/day37_2.png" style="zoom:80%;" />

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();
        //dp[i][j] word1的前i个字符转化为word2前j个字符所使用的最小操作次数
        int[][] dp = new int[len1+1][len2+1];
        for(int i = 0;i<=len1;i++){
            dp[i][0] = i;
        }
        for(int j = 0;j<=len2;j++){
            dp[0][j] = j;
        }
        
        for(int i=1;i<=len1;i++){
            for(int j=1;j<=len2;j++){
                if(word1.charAt(i-1) == word2.charAt(j-1)){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = Math.min(Math.min(dp[i][j-1],dp[i-1][j]),dp[i-1][j-1])+1;
                }
            }
        }
        return dp[len1][len2];
    }
}
```

### day38（原地旋转数组）

<img src="images/day38_1.png" style="zoom:80%;" />

idea:

1、以对角线为轴进行旋转

2、每行以中点进行旋转

```java
class Solution {
    public void rotate(int[][] matrix) {
        int m = matrix.length;
        //以对角线为轴进行翻转
        for(int i = 0;i<m;i++){
            for(int j = 0;j<m;j++){
                if(i<j){
                    int temp = 0;
                    temp = matrix[i][j];
                    matrix[i][j] = matrix[j][i];
                    matrix[j][i] = temp;
                }
            }
        }
        //对每行数据以中心为轴进行进行翻转
        int mid = m>>1;
        for(int i=0;i<m;i++){
            for(int j=0;j<mid;j++){
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][m-1-j];
                matrix[i][m-1-j] = temp;
            }
        }
    }
}
```

### day39（机器人的运动范围）

<img src="images/day39_1.png" style="zoom:80%;" />

#### DFS:

```java
class Solution {
    boolean[][] isVisited;
    public int movingCount(int m, int n, int k) {
        isVisited = new boolean[m][n];
        return dfs(0,0,m,n,k);
    }
    public int dfs(int x,int y,int m,int n,int k){
        if(x<0 || y<0 || x>=m || y>=n || isVisited[x][y]==true || (x%10+x/10+y%10+y/10)>k){
            return 0;
        }
        isVisited[x][y] = true;
        return 1+dfs(x-1,y,m,n,k)+dfs(x+1,y,m,n,k)+dfs(x,y-1,m,n,k)+dfs(x,y+1,m,n,k);
    }
}
```

#### BFS待续

```

```

### day40（括号生成--递归DFS）

<img src="images/day40_1.png" style="zoom:80%;" />

```java
class Solution {
    List<String> ans = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        dfs(n,n,"");
        return ans;
    }
    public void dfs(int left,int right,String curStr){
        if(left==0 && right==0){
            ans.add(curStr);
            return;
        }
        if(left>0){
            dfs(left-1,right,curStr+"(");
        }
        if(right>left){
            dfs(left,right-1,curStr+")");
        }
    }
}
```

### day41（反转字符串里的单词）

<img src="images/day41_1.png" style="zoom:80%;" />

```java
class Solution {
    public String reverseWords(String s) {
        String ss = s.trim();
        String[] SArrays = ss.split("\\s+");
        StringBuilder sb = new StringBuilder();
        for(int i=SArrays.length-1;i>0;i--){
            sb.append(SArrays[i]).append(" ");
        }
        sb.append(SArrays[0]);
        return sb.toString();
    }
}
```

### day42（合并K个有序链表---优先队列（堆））

<img src="images/day42_1.png" style="zoom:80%;" />

<img src="images/day42_2.png" style="zoom:80%;" />

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists==null || lists.length==0){
            return null;
        }
        int k = lists.length;
        PriorityQueue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>(){
           public int compare(ListNode o1,ListNode o2){
               //升序排序
               return (o1.val-o2.val);
           } 
        });
        
        //把每个链表的数据放入queue
        for(int i=0;i<k;i++){
            ListNode head  = lists[i];
            while(head!=null){
                queue.add(head);
                head = head.next;
            }
        }
        
        ListNode temp = new ListNode(-1);
        ListNode head = temp;
        while(!queue.isEmpty()){
            temp.next = queue.poll();
            temp = temp.next;
        }
        temp.next = null;
        return head.next;
    }
}
```

### day43（两个链表数据相加--stack）

<img src="images/day43_1.png" style="zoom:80%;" />

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();
        while(l1!=null){
            stack1.add(l1.val);
            l1 = l1.next;
        }
        
        while(l2!=null){
            stack2.add(l2.val);
            l2 = l2.next;
        }
        
        int carry = 0;
        ListNode head = null; //从尾部一个个插入链表
        while(!stack1.isEmpty() || !stack2.isEmpty() || carry>0){
            int sum = carry;
            sum += stack1.isEmpty() ? 0:stack1.pop();
            sum += stack2.isEmpty() ? 0:stack2.pop();
            ListNode node = new ListNode(sum%10);
            node.next = head;
            head = node;
            carry = sum/10;
        }
        return head;
    }
}
```

### day44（01矩阵--BFS VS 最远海洋问题）

<img src="images/day44_1.png" style="zoom:80%;" />

idea:

首先把每个源点 00 入队，然后从各个 00 同时开始一圈一圈的向 11 扩散（每个 11 都是被离它最近的 00 扩散到的 ），扩散的时候可以设置 int[][] dist 来记录距离（即扩散的层次）并同时标志是否访问过。对于本题是可以直接修改原数组 int[][] matrix 来记录距离和标志是否访问的，这里要注意先把 matrix 数组中 1 的位置设置成 -1 （设成Integer.MAX_VALUE啦，m * n啦，10000啦都行，只要是个无效的距离值来标志这个位置的 1 没有被访问过就行辣~）

```java
class Solution {
    public int[][] updateMatrix(int[][] matrix) {
        Queue<int[]> queue = new ArrayDeque<>();
        int m = matrix.length;
        int n = matrix[0].length;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(matrix[i][j]==0){
                    queue.add(new int[] {i,j});
                }else{
                    matrix[i][j] = -1;
                }
            }
        }
        
        int[] dx = {-1,1,0,0};
        int[] dy = {0,0,-1,1};
        
        while(!queue.isEmpty()){
            int[] temp = queue.poll();
            int x = temp[0];
            int y = temp[1];
            for(int k =0;k<4;k++){
                int newX = x + dx[k];
                int newY = y + dy[k];
                if(newX>=0 && newX<m && newY>=0 && newY<n && matrix[newX][newY]==-1){
                    matrix[newX][newY] = matrix[x][y] + 1;
                    queue.add(new int[] {newX,newY});
                }
            }
        }
        return matrix;
    }
}
```

### day45（合并区间）

<img src="images/day45_1.png" style="zoom:80%;" />

idea:

<img src="images/day45_2.png" style="zoom:70%;" />

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals,new Comparator<int[]>(){
           public int compare(int[] o1,int[] o2){
               return o1[0] - o2[0];
           } 
        });
        
        int[][] ans = new int[intervals.length][2];
        int count = -1;
        for(int[] interval:intervals){
            if(count==-1 || interval[0]>ans[count][1]){
                ans[++count] = interval;
            }else{
                ans[count][1] = Math.max(ans[count][1],interval[1]);
            }
            
        }
        return Arrays.copyOf(ans,count+1);
    }
}
```

### day46（跳跃数组）

<img src="images/day46_1.png" style="zoom:80%;" />

```java
class Solution {
    public boolean canJump(int[] nums) {
        int max = 0;
        for(int i = 0;i<nums.length;i++){
            if(i>max){
                return false;
            }
            max = Math.max(max,i+nums[i]); //i+nums[i] 位置i所能达到的最远距离
        }
        return max>=nums.length-1;
    }
}
```

### day47（盛最多水的容器）

<img src="images/day47_1.png" style="zoom:80%;" />

```java
class Solution {
    public int maxArea(int[] height) {
        if(height==null || height.length==0){
            return 0;
        }
        int left = 0;
        int right = height.length-1;
        int maxArea = 0;
        int area = 0;
        while(left<right){
            if(height[left]>height[right]){
                area = height[right]*(right-left);
                right--;
            }else if(height[left]<height[right]){
                area = height[left]*(right-left);
                left++;
            }else{
                area = height[left]*(right-left);
                right--;
                left++;
            }
            maxArea = Math.max(area,maxArea);
        }
        return maxArea;
    }
}
```

### day48（岛屿数量BFS）

<img src="images/day48_1.png" style="zoom:80%;" />

```java
class Solution {
    
    private class Node{
    int x = 0;
    int y = 0;
    public Node(int x,int y){
        this.x = x;
        this.y = y;
    }
    }
    
    public int m = 0;
    public int n = 0;
    public int[] dx = {-1,1,0,0};
    public int[] dy = {0,0,-1,1};
    public int numIslands(char[][] grid) {
        if(grid==null || grid.length==0){
            return 0;
        }
        
        m = grid.length;
        n = grid[0].length;
        int ans = 0;
        
        for(int i = 0;i<m;i++){
            for(int j = 0;j<n;j++){
                if(check(i,j) && grid[i][j]=='1'){
                    ans++;
                    bfs(grid,i,j);
                }
            }
        }
        
        return ans;
        
    }
    
    public boolean check(int x,int y){
        if(x<0 || x>=m || y<0 || y>=n){
            return false;
        }
        return true;
    }
    
    public void bfs(char[][] grid,int x,int y){
        Queue<Node> queue = new LinkedList<>();
        queue.add(new Node(x,y));
        grid[x][y] = 0;
        while(!queue.isEmpty()){
            Node curNode = queue.poll();
            for(int k = 0;k<4;k++){
                int next_x = curNode.x + dx[k];
                int next_y = curNode.y + dy[k];
                if(check(next_x,next_y) && grid[next_x][next_y]=='1'){
                    queue.add(new Node(next_x,next_y));
                    grid[next_x][next_y] = '0';
                }
                
            }
        }
    }
    
}

```

### day49（优美子数组---滑动窗口）

<img src="images/day49_1.png" style="zoom:80%;" />

例子：

nums = [1,1,2,1,1] k=3

<img src="images/day49_02.png" style="zoom:70%;" />

```java
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {
        int len = nums.length;
        int res = 0;
        int oddCount = 0;
        int arr[] = new int[len + 2];
        //记录第oddCount个奇数的下标
        for (int i = 0; i < len; i++) {
            if ((nums[i] & 1) == 1) {
                arr[++oddCount] = i;//第++oddCount个奇数的下标是i
            }
        }
        arr[0] = -1;//左边界
        arr[oddCount + 1] = len;//右边界

        // arr[i]是窗口左边界
        // arr[i+k-1] 是窗口右边界
        // arr[i-1]是左边的上一个奇数，在此之后到arr[i]都可选
        // arr[i+k]是右边的下一个奇数，在此之前都arr[i+k-1]都可选
        //前面可选部分长度为arr[i]-arr[i-1]
        //后面可选部分长度为arr[i+k]-arr[i+k-1]
        //总的可能数等于前后可选的组合
		//i+k<oddCount+2   i+k最大等于oddCount+1   arr[i+k] - arr[i+k-1] -->arr[oddCount]-arr[oddCount-1] 边界情况
        for (int i = 1; i + k < oddCount + 2; i++) {
            res += (arr[i] - arr[i - 1]) * (arr[i + k] - arr[i + k - 1]);
        }
        return res;
    }
}
```

### day50（二叉树的右视图）

<img src="images/day50_1.png" style="zoom:80%;" />

BFS层次遍历

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root == null){
            return res;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i=0;i<size;i++){
                TreeNode node = queue.poll();
                if(node.left!=null){
                    queue.offer(node.left);
                }
                if(node.right!=null){
                    queue.offer(node.right);
                }
                if(i==size-1){
                    res.add(node.val);
                }
            }
        }
        return res;
    }
}
```

### day51（硬币组合）

<img src="images/day51_1.png" style="zoom:80%;" />

```java
class Solution {
    public int waysToChange(int n) {
        int[] coins = {25,10,5,1};
        int[] ans = new int[n+1];
        ans[0] = 1;
        for(int coin : coins){
            for(int i=coin;i<=n;i++){
                ans[i] = (ans[i]+ans[i-coin])%1000000007;
            }
        }
        return ans[n];
    }
}
```

### day52（全排列--BFS）

<img src="images/day52_1.png" style="zoom:80%;" />

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ans = new LinkedList<>();
        Queue<List<Integer>> queue = new LinkedList<>();
        queue.add(new LinkedList<Integer>());
        
        while(!queue.isEmpty()){
            List<Integer> list = queue.poll();
            int size = list.size();
            if(size == nums.length){
                ans.add(list);
                continue;
            }
            for(int i = 0; i<=nums.length-1;i++){
                if(!list.contains(nums[i])){
                    List<Integer> temp = new LinkedList<>(list);
                    temp.add(nums[i]);
                    queue.add(temp);
                }

            }
        }
        return ans;
    }
}
```

### day53（搜索旋转排序数组--二分法查找）

<img src="images/day53_1.png" style="zoom:80%;" />

idea：

<img src="images/day53_2.png" style="zoom:80%;" />

```java
class Solution {
    public int search(int[] nums, int target) {
        if(nums==null || nums.length==0){
            return -1;
        }
        int left = 0;
        int right = nums.length-1;
        int mid = 0;
        while(left<=right){
            mid = (left+right)/2;
            if(nums[mid] == target){
                return mid;
            }
            if(nums[left]<=nums[mid]){
                if(target>=nums[left] && target<=nums[mid]){
                    right = mid-1;
                }else{
                    left = mid + 1;
                }
            }else{
                if(target>=nums[mid] && target<=nums[right]){
                    left = mid + 1;
                }else{
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

### day54（数组中数字出现的次数）

<img src="images/day54_1.png" style="zoom:80%;" />

1、如果只有一个出现一次的数字，则结果为全部数字的异或和

2、两个出现一次的数字，需要

如果我们可以把所有数字分成两组，使得：

a:两个只出现一次的数字在不同的组中。

b:相同的数字会被分到相同的组中。

要点：选择所有数字异或和的bit位中，bit=1的二进制作为mask 对所有数字进行与操作，可以实现上面两个分组的要求。

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        int sum = 0;
        for(int n:nums){
            sum ^= n;
        }
        //寻找两个不同数字的不同的bit位 即异或和bit=1
        int mask = 1;
        while((mask & sum) == 0){
            mask <<=1;
        }
        int a = 0;
        int b = 0;
        for(int n:nums){
            if((mask & n) == 0){
                a ^= n;
            }else{
                b ^= n;
            }
        }
        return new int[]{a,b};
    }
}
```

### day55 (无重复字符的最长子串----华为面试题目)

思路：
标签：滑动窗口
定义一个 map 数据结构存储 (k, v)，其中 key 值为字符，value 值为字符位置 +1，加 1 表示从字符位置后一个才开始不重复
我们定义不重复子串的开始位置为 start，结束位置为 end
随着 end 不断遍历向后，会遇到与 [start, end] 区间内字符相同的情况，此时将字符作为 key 值，获取其 value 值，并更新 start，此时 [start, end] 区间内不存在重复字符
无论是否更新 start，都会更新其 map 数据结构和结果 ans。

```java
    public static int numOfChars2(String s){
        int n = s.length();
        int ans = 0;
        Map<Character,Integer> map = new HashMap<>();
        for(int start = 0,end = 0; end<n; end++){
            char c = s.charAt(end);
            if(map.containsKey(c)){
                start = Math.max(map.get(c),start);
            }
            ans = Math.max(ans,end-start+1);
            map.put(s.charAt(end),end+1);
        }
        return ans;
    }
```

### day56（重构二叉树）

<img src="images/day56_1.png" style="zoom:80%;" />

**递归解析：**
**递推参数**： 前序遍历中根节点的索引pre_root、中序遍历左边界in_left、中序遍历右边界in_right。
**终止条件**： 当 in_left > in_right ，子树中序遍历为空，说明已经越过叶子节点，此时返回 nullnull 。
递推工作：
**建立根节点**root： 值为前序遍历中索引为pre_root的节点值。
	搜索根节点root在中序遍历的索引i： 为了提升搜索效率，本题解使用哈希表 dic 预存储中序遍历的值与索引的	映射关系，每次搜索的时间复杂度为 O(1)O(1)。
    构建根节点root的左子树和右子树： 通过调用 recur() 方法开启下一层递归。
**左子树**： 根节点索引为 pre_root + 1 ，中序遍历的左右边界分别为 in_left 和 i - 1。
**右子树**： 根节点索引为 i - in_left + pre_root + 1（即：根节点索引 + 左子树长度 + 1），中序遍历的左右边界分别为 i + 1 和 in_right。
**返回值**： 返回 root，含义是当前递归层级建立的根节点 root 为上一递归层级的根节点的左或右子节点。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    HashMap<Integer,Integer> map = new HashMap<>();
    int[] OriPreOrder;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        OriPreOrder = preorder;
        for(int i=0;i<preorder.length;i++){
            map.put(inorder[i],i);
        }
        return recur(0,0,inorder.length-1);
    }
    
    public TreeNode recur(int pre_root,int in_left,int in_right){
        if(in_left>in_right){
            return null;
        }
        TreeNode root = new TreeNode(OriPreOrder[pre_root]);
        int i = map.get(OriPreOrder[pre_root]);
        root.left = recur(pre_root+1,in_left,i-1);
        root.right = recur(pre_root+(i-in_left)+1,i+1,in_right);
        return root;
    }
}
```

### day57（Z型遍历二叉树）

```java
class Node {
    char value;  //数据域
    Node left;  //左孩子节点
    Node right;   //右孩子节点

    //}
    public Node(char value) {
        this.value = value;
    }
}

public  ArrayList<ArrayList<Character>> zigzagOrder(Node root) {
            int level = 1;   //指示当前遍历的层数
            Stack<Node> stack1 = new Stack<>();  //栈1存奇数节点
            stack1.push(root);   //将根节点入栈
            Stack<Node> stack2 = new Stack<>();  //栈2存偶数节点
            ArrayList<ArrayList<Character>> list = new ArrayList<>();
            while (!stack1.empty() || !stack2.empty()) {
                if (level % 2 != 0) {  //奇数层,该层为奇数层，叶子节点从右向左入栈，所以该层的叶子节点应入偶数栈
                    ArrayList<Character> t = new ArrayList<>();
                    while (!stack1.empty()) {
                        Node cur = stack1.pop();
                        if (cur != null) {
                            t.add(cur.value);
                            stack2.push(cur.left);
                            stack2.push(cur.right);
                        }
                    }
                    if (!t.isEmpty()) {
                        list.add(t);
                        level++;
                    }
                } else {
                    ArrayList<Character> t = new ArrayList<>();
                    while (!stack2.empty()) {
                        Node cur = stack2.pop();
                        if (cur != null) {
                            t.add(cur.value);
                            stack1.push(cur.right);
                            stack1.push(cur.left);
                        }
                    }
                    if (!t.isEmpty()) {
                        list.add(t);
                        level++;
                    }
                }

            }
            return list;
        }
```

### day58（对称二叉树）

<img src="images/day58.png" style="zoom:70%;" />

```java
//递归
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    public boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }
}



//迭代
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    public boolean check(TreeNode u, TreeNode v) {
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(u);
        q.offer(v);
        while (!q.isEmpty()) {
            u = q.poll();
            v = q.poll();
            if (u == null && v == null) {
                continue;
            }
            if ((u == null || v == null) || (u.val != v.val)) {
                return false;
            }

            q.offer(u.left);
            q.offer(v.right);

            q.offer(u.right);
            q.offer(v.left);
        }
        return true;
    }
}

```

### day59（多少质数的和等于输入的这个整数）

给定一个正整数，编写程序计算有多少对质数的和等于输入的这个正整数，并输出结果。输入值小于1000。
如，输入为10, 程序应该输出结果为2。（共有两对质数的和为10,分别为(5,5),(3,7)） 

##### 输入描述:

```
输入包括一个整数n,(3 ≤ n < 1000)
```

```java
package ICBC;

import java.util.Scanner;

public class Test11 {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        while (scanner.hasNext()){
            int n = scanner.nextInt();
            int count = 0;
            for (int j=1;j<=n;j++){
                if(isPrime(j)){
                    System.out.print(j+" ");
                }
            }
            for(int i=2;i<=n/2;i++){
                if(isPrime(i) && isPrime(n-i)){
                    count ++;
                }
            }
            System.out.println(count);
        }
    }

    public static boolean isPrime(int num){
        if(num<=1){   //质数：大于1的自然数中只能被1和自身整除的数
            return false;
        }
        if(num==2){
            return true;
        }

        boolean flag = true;
        for(int i =2;i<=Math.sqrt(num);i++){
            if(num%i==0){
                flag = false;
                break;
            }
        }
        if(flag){
            return true;
        }else{
            return false;
        }
    }
}

```

### day60（移动0到末尾）

<img src="images/day60.png" style="zoom:80%;" />

```java
    public static int[] moveZeros(int[] arr){
        if(arr == null){
            return null;
        }

        int j = 0;
        for(int i=0;i<arr.length;i++){
            if(arr[i] != 0){
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j++] = temp;
            }
        }
        return arr;
    }
```

### day61（死锁）

```java
package ByteDance;

/**
 * 线程死锁
 */
public class Lock {
    private static Object resource1 = new Object();
    private static Object resource2 = new Object();

    public static void main(String[] args) {
        new Thread(()->{
            synchronized(resource1){
                System.out.println(Thread.currentThread()+"get resource1");
                try{
                    Thread.sleep(1000);

                }catch (InterruptedException e){
                    e.printStackTrace();
                }
                System.out.println(Thread.currentThread()+"waitting get resource2");
                synchronized (resource2){
                    System.out.println(Thread.currentThread()+"get resource2");
                }
            }
        },"线程1").start();

        new Thread(()->{
            synchronized (resource2){
                System.out.println(Thread.currentThread()+"get resource2");
                try{
                    Thread.sleep(1000);

                }catch (InterruptedException e){
                    e.printStackTrace();
                }
                System.out.println(Thread.currentThread()+"waitting get resource1");
                synchronized(resource1){
                    System.out.println(Thread.currentThread()+"get resource1");
                }
            }
        },"线程2").start();
    }
    
    //解决死锁方法---------TODO--------
            new Thread(() -> {
            synchronized (resource1) {   //和线程1获取资源顺序同步
                System.out.println(Thread.currentThread() + "get resource1");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println(Thread.currentThread() + "waiting get resource2");
                synchronized (resource2) {  //和线程1获取资源顺序同步
                    System.out.println(Thread.currentThread() + "get resource2");
                }
            }
        }, "线程 2").start();
}
```

### day62（LRU算法实现）

<img src="images/day62.png" style="zoom:67%;" />

#### 1：LinkedHashMap实现方法(:smile:)

```java
public class LRU {
    //把最近使用的放尾巴就好了，删除的时候就删除头。
    private int capacity;
    private Map<Integer,Integer> map;

    public LRU(int capacity){
        this.capacity = capacity;
        map = new LinkedHashMap<>();
    }

    public int get(int key){
        if(!map.containsKey(key)){
            return -1;
        }
        int val = map.remove(key);
        map.put(key,val);
        return val;
    }
    
    public void put(int key,int val){
        if(map.containsKey(key)){
            map.remove(key);
            map.put(key,val);
            return;
        }
        map.put(key,val);
        if(map.size()>capacity){
            //最近使用的在尾部  所以删除头部
            map.remove(map.entrySet().iterator().next().getKey());  
        }
    }
```

#### 2：HashMap+双向链表

```java
package ByteDance;

import java.util.HashMap;
import java.util.Map;

/**
 * HashMap+双向链表实现
 */
public class LRULinked {
    private class DoubleLinkedNode{
        int key;
        int val;
        DoubleLinkedNode pre;
        DoubleLinkedNode next;

        DoubleLinkedNode(int key,int val){
            this.key = key;
            this.val = val;
            pre = null;
            next = null;
        }
    }

    private int capacity;
    private Map<Integer,DoubleLinkedNode> map;
    private DoubleLinkedNode head;
    private DoubleLinkedNode tail;

    public LRULinked(int capacity){
        this.capacity = capacity;
        map = new HashMap<>();
        head = new DoubleLinkedNode(-1,-1);
        tail = new DoubleLinkedNode(-1,-1);

        head.next = tail;
        tail.pre = head;
    }

    public int get(int key){
        if(!map.containsKey(key)){
            return -1;
        }
        DoubleLinkedNode node = map.get(key);
        //先把这个节点删除，再接到尾部
        node.pre.next = node.next;
        node.next.pre = node.pre;
        moveToTail(node);

        return node.val;
    }

    public void put(int key, int value) {
        //直接调用这边的get方法，如果存在，它会在get内部被移动到尾巴，不用再移动一遍,直接修改值即可
        if (get(key) != -1) {
            map.get(key).val = value;
            return;
        }
        //不存在，new一个出来,如果超出容量，把头去掉
        DoubleLinkedNode node = new DoubleLinkedNode(key, value);
        map.put(key, node);
        moveToTail(node);

        if (map.size() > capacity) {
            map.remove(head.next.key);
            head.next = head.next.next;
            head.next.pre = head;
        }
    }
    private void moveToTail(DoubleLinkedNode node) {
        node.pre = tail.pre;
        tail.pre = node;
        node.pre.next = node;
        node.next = tail;
    }
}

```

### day63（比较版本大小）

```java
/**
 * 比较版本号大小
 * “1.0.1”和“1”，返回1
 * “1.1.1”和“1.8.2”，返回-1
 * “1.0.1”和“1.0.01”，返回1
 */
public class CompareVersion {

    public static void main(String[] args) {
        String s1 = "1.0.1";
        String s2 = "1.1.1";
        System.out.println(compare(s1,s2));
    }

    public static int compare(String str1,String str2){
        char[] chars1 = str1.toCharArray();
        char[] chars2 = str2.toCharArray();
        int len1 = chars1.length;
        int len2 = chars2.length;
        int len = Math.min(len1,len2);

        for(int i =0;i<len;i++){
            if(chars1[i]==chars2[i]){
                continue;
            }

            if(chars1[i]>chars2[i]){
                return 1;
            }else{
                return -1;
            }
        }
        return 1;
    }
}
```

### day64（三个线程循环打印数字--字节面试）

```java
package ByteDance;

public class Test05 {
    public static void main(String[] args) throws InterruptedException{
        Thread t1 = new Thread(new MyThread1(0));
        Thread t2 = new Thread(new MyThread1(1));
        Thread t3 = new Thread(new MyThread1(2));
        t1.start();
        t2.start();
        t3.start();
        t1.join();
        t2.join();
        t3.join();

    }
}

class MyThread1 implements Runnable{
    private static Object lock = new Object();
    private static int count = 0;
    int no;

    MyThread1(int no){
        this.no = no;
    }

    @Override
    public void run() {
        while (true){
            synchronized (lock){
                if(count>100){
                    break;
                }
                if(count%3==this.no){
                    System.out.println(this.no+"----->"+count);
                    count++;
                }else{
                    try{
                        lock.wait();
                    }catch (InterruptedException e){
                        e.printStackTrace();
                    }
                }
                lock.notifyAll();
            }
        }
    }
}

```

### day65（和最大的连续子数组）

```java
package ByteDance;

public class Test06 {
    public static void main(String[] args) {
        int[] arr = new int[]{1,-2,3,10,-4,7,2,-5};
        System.out.println(findMaxSum(arr));
    }

    public static int findMaxSum(int[] arr) {
        if(arr == null || arr.length == 0){
            throw new IllegalArgumentException("数组不合法");
        }

        int curSum = 0;
        int maxSum = arr[0];

        //找到下标
        int start = 0;
        int end = 0;
        for(int i=0;i<arr.length;i++){
//            curSum = (arr[i]>arr[i]+curSum)? arr[i]:curSum+arr[i];
//            maxSum = Math.max(curSum,maxSum);
            if(arr[i]>arr[i]+curSum){
                start = i;
                curSum = arr[i];
            }else{
                end = i-1;
                curSum = arr[i]+curSum;
            }
            maxSum = Math.max(curSum,maxSum);
        }
        System.out.println("连续数组的下标为："+start+"   "+end);
        return maxSum;
    }
}

```

### day66（二叉树问题）

**1：层次遍历----Queue**

```java
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if(root == null)
            return result;
        LinkedList<Integer> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            result.add(node.val);
            if(node.left != null){
                queue.add(node.left);
            }
            if(node.right != null){
                queue.add(node.right);
            }
        }
        return result;
    }
```

2：**前序遍历**----->**stack**

```java
public class TreeNode{
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int val){
        this.val = val;
    }
}
//非递归
public void preOrder(TreeNode root){
    if(root == null) return;
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    while(!stack.isEmpty()){
        TreeNode node = stack.pop();
        System.out.print(node.val);
        //因为栈是先进后出，所以先压右孩子，再压左孩子
        if(node.right != null)
            stack.push(node.right);
        if(node.left != null)
            stack.push(node.left);
    } 
}
//递归
public void preOrder(TreeNode root){
    if(root == null) return;
    System.out.print(root.val);
    preOrder(root.let);
    preOrder(root.right);
}
```

3：**后序遍历**

```java
public void postOrder(TreeNode root){
    if(root == null) return;
    Stack<TreeNode> s1 = new Stack<>();
    Stack<TreeNode> s2 = new Stack<>();
    s1.push(root);
    while(!s1.isEmpty()){
        TreeNode node = s1.pop();
        s2.push(node);
        if(node.left != null)
           s1.push(node.left);
       if(node.right != null)
           s1.push(node.right);
    }
    while(!s2.isEmpty())
        System.out.print(s2.pop().val + " ");
}
```

### Bilibili笔试题目：

<img src="images/day63.png" style="zoom:80%;" />

```java
class Solution {
    public int findMaxConsecutiveOnes(int[] nums) {
        int count = 0;
        int MaxCount = 0;
        for(int i=0;i<nums.length;i++){
            if(nums[i] == 1){
                count ++;
            }else{
                MaxCount = Math.max(MaxCount,count);
                count = 0;
            }
        }
        return Math.max(MaxCount,count);
    }
}
```

<img src="images/day63_1.png" style="zoom:80%;" />

<img src="images/day63_2.png" style="zoom:80%;" />

```java
public static int longestOnes(int[] nums,int k){
    int ans = 0;
    for(int count =0,l=0,r=0;r<nums.length;++r){
        if(nums[r] == 1){
            ++count;
        }
        while (r-l+1-count>k){
            if(nums[l++] == 1){
                --count;
            }
        }
        ans = Math.max(ans,r-l+1);
    }
    return ans;
}
```

<img src="images/day64.png" style="zoom:80%;" />

解题思路：计算不同代码块的数量。

```java
public static int GetFragment (String str) {
    // write code here
    int len = str.length();
    int count = 1;
    for(int i=0;i<len-1;i++){
        if(str.charAt(i)!=str.charAt(i+1))
            count ++;
    }
    return len/count;
}
```

