`timescale 1ns / 1ps

module bram_wrapper#(
    parameter READ_LATENCY = 3,
    parameter ADDR_WIDTH = 15,
    parameter DATA_WIDTH = 31
)(
input  wire clk,
input  wire rst,
input  wire [ADDR_WIDTH-1 : 0] addr,
input  wire en,
input  wire we,
input  wire [DATA_WIDTH-1 : 0] din,
output wire [DATA_WIDTH-1 : 0] dout,
output wire valid
);

assign valid = valid_to_ext;

enum { IDLE, WAIT, READ } state, next_state;

reg [2:0] wait_cnt, wait_cnt_next;
reg en_to_bram, valid_to_ext;

// State Update
always @(posedge clk, negedge rst) begin
    if(!rst) state <= IDLE;
    else state <= next_state;
end

// Set Next State 
always @(*) begin
    next_state = state;
    case(state)
        IDLE : begin
                    if(en == 1'b1 && we == 1'b0) begin
                        if(READ_LATENCY ==1) next_state = READ;
                        else next_state = WAIT;
                    end
                end
        WAIT : next_state = (wait_cnt == READ_LATENCY-1) ? READ : WAIT;
        READ : next_state = IDLE;
    endcase
end
// Set Output Signal
always @(*) begin
    en_to_bram = en;
    valid_to_ext = 0;
    case(state)
        IDLE : begin
               en_to_bram = en;
               valid_to_ext = 0;
               end
        WAIT : begin
               en_to_bram = 1;
               valid_to_ext = 0;
               end
        READ : begin
               en_to_bram = 0;
               valid_to_ext = 1;
               end
    endcase
end

always @(posedge clk, negedge rst) begin
    if(!rst) wait_cnt <= 0;
    else wait_cnt <= wait_cnt_next;
end

always @(*) begin
    wait_cnt_next = wait_cnt;
    case(state)
        IDLE : wait_cnt_next = 2'b1;
        WAIT : wait_cnt_next = wait_cnt + 1'b1;
        default : wait_cnt_next = 0;
    endcase
end


endmodule